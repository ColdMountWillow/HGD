"""
训练脚本：Hypergraph-guided Diffusion for Unpaired Virtual Stain Translation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from config import Config, load_config
from models import UNet, Diffusion
from hypergraph import construct_hyperedges
from hypergraph.hgnn import HypergraphConv
from losses import DiffusionLoss, HypergraphContrastiveLoss
from data.dataset import UnpairedStainDataset


class HypergraphDiffusionModel(nn.Module):
    """
    完整的 Hypergraph-guided Diffusion 模型
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # U-Net
        self.unet = UNet(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            base_channels=config.base_channels,
            channel_multipliers=config.channel_multipliers,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=config.attention_resolutions,
            condition_channels=config.in_channels,
            extract_features=True
        )
        
        # Hypergraph Conv
        self.hypergraph_conv = HypergraphConv(
            in_channels=config.feature_dim,
            out_channels=config.feature_dim
        )
        
        # Diffusion
        self.diffusion = Diffusion(
            num_timesteps=config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            device=config.device
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：预测噪声
        
        Args:
            x: 噪声图像 [B, C, H, W]
            timestep: 时间步 [B]
            condition: 条件图像 [B, C, H, W]
        
        Returns:
            predicted_noise: 预测的噪声 [B, C, H, W]
        """
        return self.unet(x, timestep, condition)
    
    def extract_and_process_features(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        提取并处理 patch features（用于 hypergraph loss）
        
        Args:
            x: 噪声图像 [B, C, H, W]
            timestep: 时间步 [B]
            condition: 条件图像 [B, C, H, W]
        
        Returns:
            processed_features: 处理后的 features [B, num_patches, feature_dim]
        """
        # 提取 patch features
        patch_features = self.unet.extract_patch_features(
            x, timestep, condition, num_patches=self.config.patch_size
        )  # [B, num_patches, feature_dim]
        
        # 构建 hypergraph
        hyperedge_matrix, point_hyperedge_index, hyperedge_centers = construct_hyperedges(
            patch_features,
            num_clusters=self.config.num_hyperedges,
            threshold=self.config.hyperedge_threshold
        )
        
        # Hypergraph convolution
        # Reshape 为 [B, C, N, 1] 格式
        patch_features_reshaped = patch_features.permute(0, 2, 1).unsqueeze(-1)
        # [B, feature_dim, num_patches, 1]
        
        processed_features = self.hypergraph_conv(
            patch_features_reshaped,
            hyperedge_matrix,
            point_hyperedge_index,
            hyperedge_centers
        )  # [B, feature_dim, num_patches, 1]
        
        # Reshape 回 [B, num_patches, feature_dim]
        processed_features = processed_features.squeeze(-1).permute(0, 2, 1)
        
        return processed_features


def train_epoch(
    model: HypergraphDiffusionModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    diffusion_loss_fn: DiffusionLoss,
    hypergraph_loss_fn: HypergraphContrastiveLoss,
    config: Config,
    epoch: int
) -> dict:
    """
    训练一个 epoch
    
    Returns:
        losses: 损失字典
    """
    model.train()
    total_diffusion_loss = 0.0
    total_hypergraph_loss = 0.0
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (source_images, target_images) in enumerate(pbar):
        source_images = source_images.to(config.device)
        target_images = target_images.to(config.device)
        
        B = source_images.shape[0]
        
        # ========== Diffusion Loss ==========
        # 随机采样时间步
        t = torch.randint(0, config.num_timesteps, (B,), device=config.device).long()
        
        # 生成噪声
        noise = torch.randn_like(target_images)
        
        # 前向扩散：x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        x_t = model.diffusion.q_sample(target_images, t, noise)
        
        # 预测噪声（以 source 为条件）
        predicted_noise = model(x_t, t, source_images)
        
        # Diffusion loss
        diffusion_loss = diffusion_loss_fn(predicted_noise, noise)
        
        # ========== Hypergraph Loss ==========
        # 从 source 和 target 分别提取 features
        # 使用相同的噪声图像和时间步，但不同的条件
        source_features = model.extract_and_process_features(
            x_t, t, source_images
        )  # [B, num_patches, feature_dim]
        
        # 对于 target，我们使用 target 图像本身作为条件（在训练时）
        # 或者使用 source 作为条件（更符合实际应用）
        target_features = model.extract_and_process_features(
            x_t, t, source_images  # 使用 source 作为条件，但目标是 target domain
        )
        
        # 为了更好的对比，我们可以从 target 图像提取 features（不经过 diffusion）
        # 这里简化处理：使用 target 图像直接提取 features（不添加噪声）
        with torch.no_grad():
            # 从 target 图像提取特征（作为参考）
            t_zero = torch.zeros(B, dtype=torch.long, device=config.device)
            target_ref_features = model.unet.extract_patch_features(
                target_images, t_zero, target_images, num_patches=config.patch_size
            )
        
        # Hypergraph contrastive loss
        hypergraph_loss = hypergraph_loss_fn(source_features, target_ref_features)
        
        # ========== Total Loss ==========
        total_batch_loss = (
            config.diffusion_loss_weight * diffusion_loss +
            config.hypergraph_loss_weight * hypergraph_loss
        )
        
        # 反向传播
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()
        
        # 累计损失
        total_diffusion_loss += diffusion_loss.item()
        total_hypergraph_loss += hypergraph_loss.item()
        total_loss += total_batch_loss.item()
        
        # 更新进度条
        pbar.set_postfix({
            'diff_loss': f'{diffusion_loss.item():.4f}',
            'hg_loss': f'{hypergraph_loss.item():.4f}',
            'total': f'{total_batch_loss.item():.4f}'
        })
    
    return {
        'diffusion_loss': total_diffusion_loss / len(dataloader),
        'hypergraph_loss': total_hypergraph_loss / len(dataloader),
        'total_loss': total_loss / len(dataloader)
    }


def sample_images(
    model: HypergraphDiffusionModel,
    source_images: torch.Tensor,
    config: Config,
    num_samples: int = 4
) -> torch.Tensor:
    """
    采样生成图像
    
    Args:
        model: 训练好的模型
        source_images: Source 图像 [B, C, H, W]
        config: 配置
        num_samples: 采样数量
    
    Returns:
        generated_images: 生成的图像 [B, C, H, W]
    """
    model.eval()
    
    with torch.no_grad():
        # 从纯噪声开始采样
        shape = (num_samples, config.out_channels, config.image_size, config.image_size)
        generated_images = model.diffusion.p_sample_loop(
            model.unet,
            shape,
            condition=source_images[:num_samples]
        )
    
    return generated_images


def save_samples(
    source_images: torch.Tensor,
    generated_images: torch.Tensor,
    save_path: str,
    epoch: int
):
    """
    保存采样结果
    """
    # 反归一化：从 [-1, 1] 到 [0, 1]
    source_images = (source_images + 1) / 2
    generated_images = (generated_images + 1) / 2
    
    # 限制到 [0, 1]
    source_images = torch.clamp(source_images, 0, 1)
    generated_images = torch.clamp(generated_images, 0, 1)
    
    # 创建对比图
    fig, axes = plt.subplots(2, source_images.shape[0], figsize=(4 * source_images.shape[0], 8))
    if source_images.shape[0] == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(source_images.shape[0]):
        # Source
        axes[0, i].imshow(source_images[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title(f"Source {i+1}")
        axes[0, i].axis('off')
        
        # Generated
        axes[1, i].imshow(generated_images[i].permute(1, 2, 0).cpu().numpy())
        axes[1, i].set_title(f"Generated {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"samples_epoch_{epoch}.png"))
    plt.close()


def main():
    """主训练函数"""
    # 加载配置
    config = load_config()
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 创建目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # 设备
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = device
    print(f"Using device: {device}")
    
    # 数据集
    source_dir = os.path.join(config.data_root, config.source_domain)
    target_dir = os.path.join(config.data_root, config.target_domain)
    
    dataset = UnpairedStainDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        image_size=config.image_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # 模型
    model = HypergraphDiffusionModel(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数
    diffusion_loss_fn = DiffusionLoss()
    hypergraph_loss_fn = HypergraphContrastiveLoss(temperature=config.temperature)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 训练循环
    print("\nStarting training...")
    for epoch in range(1, config.num_epochs + 1):
        # 训练
        losses = train_epoch(
            model, dataloader, optimizer,
            diffusion_loss_fn, hypergraph_loss_fn,
            config, epoch
        )
        
        print(f"\nEpoch {epoch}/{config.num_epochs}:")
        print(f"  Diffusion Loss: {losses['diffusion_loss']:.4f}")
        print(f"  Hypergraph Loss: {losses['hypergraph_loss']:.4f}")
        print(f"  Total Loss: {losses['total_loss']:.4f}")
        
        # 采样
        if epoch % config.sample_freq == 0:
            print("Sampling...")
            sample_batch = next(iter(dataloader))
            source_samples = sample_batch[0][:4].to(device)
            
            generated_samples = sample_images(model, source_samples, config, num_samples=4)
            save_samples(source_samples, generated_samples, config.sample_dir, epoch)
            print(f"Saved samples to {config.sample_dir}")
        
        # 保存检查点
        if epoch % config.save_freq == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

