"""
Gaussian Diffusion 核心实现
支持 DDPM 训练 和 DDIM 采样
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Callable


def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """线性 beta schedule"""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule"""
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Process
    
    支持:
    - DDPM 训练 (预测噪声)
    - DDIM 采样 (加速采样)
    - 返回中间特征 (用于 hypergraph loss)
    """
    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        sampling_timesteps: int = 50,
        ddim_eta: float = 0.0
    ):
        super().__init__()
        
        self.model = model
        self.num_timesteps = num_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.ddim_eta = ddim_eta
        
        # Beta schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # 预计算各种系数
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 注册为 buffer (不参与训练，但会保存到 checkpoint)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # 前向扩散 q(x_t | x_0) 的系数
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # 后验 q(x_{t-1} | x_t, x_0) 的系数
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """从 schedule 中提取对应 timestep 的值"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(
        self, 
        x_0: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向扩散: q(x_t | x_0)
        
        Args:
            x_0: (B, C, H, W) 原始图像
            t: (B,) timesteps
            noise: (B, C, H, W) 可选，预定义噪声
        
        Returns:
            x_t: (B, C, H, W) 加噪后的图像
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
    
    def p_losses(
        self, 
        x_0: torch.Tensor, 
        cond: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        domain_id: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        计算训练损失
        
        Args:
            x_0: (B, C, H, W) target domain 图像
            cond: (B, C, H, W) source domain 图像 (condition)
            t: (B,) timesteps, 如果 None 则随机采样
            domain_id: (B,) domain embedding id
            noise: (B, C, H, W) 可选噪声
        
        Returns:
            loss: scalar, MSE loss
            noise_pred: (B, C, H, W) 预测的噪声
            mid_features: (B, C_mid, H_mid, W_mid) 中间特征
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # 随机采样 timestep
        if t is None:
            t = torch.randint(0, self.num_timesteps, (B,), device=device, dtype=torch.long)
        
        # 采样噪声
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # 前向扩散得到 x_t
        x_t = self.q_sample(x_0, t, noise)
        
        # 预测噪声
        noise_pred, mid_features = self.model(x_t, t, cond, domain_id)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss, noise_pred, mid_features
    
    def predict_x0_from_noise(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        noise: torch.Tensor
    ) -> torch.Tensor:
        """从 x_t 和预测的噪声恢复 x_0"""
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        return (x_t - sqrt_one_minus_alpha_cumprod * noise) / sqrt_alpha_cumprod
    
    @torch.no_grad()
    def p_sample(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        cond: torch.Tensor,
        domain_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        DDPM 单步采样: p(x_{t-1} | x_t)
        """
        noise_pred, _ = self.model(x_t, t, cond, domain_id)
        
        # 预测 x_0
        x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # 计算后验均值
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0_pred +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        
        # 添加噪声 (t > 0 时)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise
    
    @torch.no_grad()
    def ddim_sample_step(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        t_prev: torch.Tensor,
        cond: torch.Tensor,
        domain_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        DDIM 单步采样
        """
        noise_pred, _ = self.model(x_t, t, cond, domain_id)
        
        # 当前和上一步的 alpha
        alpha_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_t_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)
        
        # 预测 x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # DDIM 方向
        sigma = self.ddim_eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
        
        # 预测 x_{t-1}
        dir_xt = torch.sqrt(1 - alpha_t_prev - sigma ** 2) * noise_pred
        noise = torch.randn_like(x_t) if self.ddim_eta > 0 else 0
        
        x_t_prev = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt + sigma * noise
        
        return x_t_prev
    
    @torch.no_grad()
    def sample(
        self, 
        cond: torch.Tensor,
        domain_id: Optional[torch.Tensor] = None,
        use_ddim: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        从噪声采样生成图像
        
        Args:
            cond: (B, C, H, W) condition image
            domain_id: (B,) domain id
            use_ddim: 是否使用 DDIM 加速采样
            progress_callback: 进度回调函数
        
        Returns:
            x_0: (B, C, H, W) 生成的图像
        """
        device = cond.device
        B = cond.shape[0]
        shape = cond.shape
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        if use_ddim:
            # DDIM 采样
            timesteps = torch.linspace(
                self.num_timesteps - 1, 0, self.sampling_timesteps, 
                dtype=torch.long, device=device
            )
            
            for i, t in enumerate(timesteps):
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)
                t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else torch.zeros_like(t_batch)
                t_prev_batch = torch.full((B,), t_prev, device=device, dtype=torch.long)
                
                x = self.ddim_sample_step(x, t_batch, t_prev_batch, cond, domain_id)
                
                if progress_callback:
                    progress_callback(i, len(timesteps))
        else:
            # DDPM 采样
            for t in reversed(range(self.num_timesteps)):
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)
                x = self.p_sample(x, t_batch, cond, domain_id)
                
                if progress_callback:
                    progress_callback(self.num_timesteps - t, self.num_timesteps)
        
        return x
    
    def forward(
        self, 
        x_0: torch.Tensor, 
        cond: torch.Tensor,
        domain_id: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        训练 forward pass
        
        Returns:
            loss, noise_pred, mid_features
        """
        return self.p_losses(x_0, cond, domain_id=domain_id)


# 测试代码
if __name__ == "__main__":
    from unet import ConditionalUNet
    
    # 创建模型
    unet = ConditionalUNet(
        in_channels=3,
        out_channels=3,
        model_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        image_size=64
    )
    
    diffusion = GaussianDiffusion(
        model=unet,
        num_timesteps=100,
        sampling_timesteps=10
    )
    
    # 测试训练
    B = 2
    x_0 = torch.randn(B, 3, 64, 64)
    cond = torch.randn(B, 3, 64, 64)
    
    loss, noise_pred, mid_feat = diffusion(x_0, cond)
    print(f"Training loss: {loss.item():.4f}")
    print(f"Noise pred shape: {noise_pred.shape}")
    print(f"Mid feature shape: {mid_feat.shape}")
    
    # 测试采样
    print("\nTesting sampling...")
    sample = diffusion.sample(cond, use_ddim=True)
    print(f"Sample shape: {sample.shape}")

