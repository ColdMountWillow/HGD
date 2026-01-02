"""
Diffusion 调度器和采样器
基于 DDPM 实现
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class Diffusion:
    """
    Diffusion 模型调度器
    
    实现 DDPM 的前向扩散过程和噪声调度
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        device: str = "cuda"
    ):
        """
        Args:
            num_timesteps: 扩散时间步数
            beta_start: beta 起始值
            beta_end: beta 结束值
            beta_schedule: 调度方式 ("linear" 或 "cosine")
            device: 设备
        """
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 计算 beta 调度
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif beta_schedule == "cosine":
            # Cosine schedule (简化版)
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps, device=device)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # 计算 alpha 相关参数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        
        # 用于采样的参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向扩散过程：q(x_t | x_0)
        
        Args:
            x_start: 原始图像 [B, C, H, W]
            t: 时间步 [B]
            noise: 可选的外部噪声
        
        Returns:
            x_t: 加噪后的图像 [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        单步采样：p(x_{t-1} | x_t)
        
        Args:
            model: U-Net 模型
            x: 当前噪声图像 [B, C, H, W]
            t: 时间步 [B]
            condition: 条件图像 [B, C, H, W]
        
        Returns:
            x_prev: 去噪后的图像 [B, C, H, W]
        """
        # 预测噪声
        predicted_noise = model(x, t, condition)
        
        # 计算 x_0 的预测
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        pred_x_start = (x - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        
        # 计算 x_{t-1}
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        posterior_mean = (
            torch.sqrt(alpha_cumprod_prev_t) * self.betas[t].view(-1, 1, 1, 1) / (1.0 - alpha_cumprod_t) * pred_x_start
            + torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t) * x
        )
        
        posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
        
        return posterior_mean + torch.sqrt(posterior_variance_t) * noise
    
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple,
        condition: Optional[torch.Tensor] = None,
        return_all_timesteps: bool = False
    ) -> torch.Tensor:
        """
        DDPM 采样循环
        
        Args:
            model: U-Net 模型
            shape: 输出图像形状 (B, C, H, W)
            condition: 条件图像 [B, C, H, W]
            return_all_timesteps: 是否返回所有时间步
        
        Returns:
            x: 生成的图像 [B, C, H, W]
        """
        device = next(model.parameters()).device
        b = shape[0]
        
        # 从纯噪声开始
        img = torch.randn(shape, device=device)
        imgs = [img] if return_all_timesteps else None
        
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, condition)
            if return_all_timesteps:
                imgs.append(img)
        
        return img if not return_all_timesteps else torch.stack(imgs)

