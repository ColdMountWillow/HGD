"""
使用示例：展示如何训练和使用 Hypergraph-guided Diffusion 模型
"""
import torch
from config import Config
from models import UNet, Diffusion
from hypergraph import construct_hyperedges
from hypergraph.hgnn import HypergraphConv
from losses import DiffusionLoss, HypergraphContrastiveLoss


def example_training_step():
    """示例：单个训练步骤"""
    print("=" * 60)
    print("示例：训练步骤")
    print("=" * 60)
    
    # 配置
    config = Config()
    config.batch_size = 2
    config.image_size = 256
    config.num_timesteps = 1000
    config.patch_size = 64
    config.num_hyperedges = 9
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(config.device)
    
    # 创建模型
    unet = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        condition_channels=3
    ).to(device)
    
    diffusion = Diffusion(
        num_timesteps=config.num_timesteps,
        device=config.device
    )
    
    # 创建损失函数
    diffusion_loss_fn = DiffusionLoss()
    hypergraph_loss_fn = HypergraphContrastiveLoss()
    
    # 模拟数据
    source_images = torch.randn(config.batch_size, 3, config.image_size, config.image_size).to(device)
    target_images = torch.randn(config.batch_size, 3, config.image_size, config.image_size).to(device)
    
    # 随机时间步
    t = torch.randint(0, config.num_timesteps, (config.batch_size,), device=device).long()
    
    # 生成噪声
    noise = torch.randn_like(target_images)
    
    # Forward diffusion
    x_t = diffusion.q_sample(target_images, t, noise)
    print(f"x_t shape: {x_t.shape}")
    
    # 预测噪声
    predicted_noise = unet(x_t, t, source_images)
    print(f"predicted_noise shape: {predicted_noise.shape}")
    
    # Diffusion loss
    diff_loss = diffusion_loss_fn(predicted_noise, noise)
    print(f"Diffusion loss: {diff_loss.item():.4f}")
    
    # 提取 patch features
    source_features = unet.extract_patch_features(
        x_t, t, source_images, num_patches=config.patch_size
    )
    print(f"Source patch features shape: {source_features.shape}")
    
    # 构建 hypergraph
    hyperedge_matrix, point_hyperedge_index, hyperedge_centers = construct_hyperedges(
        source_features,
        num_clusters=config.num_hyperedges,
        threshold=0.15
    )
    print(f"Hyperedge matrix shape: {hyperedge_matrix.shape}")
    print(f"Point-hyperedge index shape: {point_hyperedge_index.shape}")
    print(f"Hyperedge centers shape: {hyperedge_centers.shape}")
    
    # Hypergraph convolution
    hgnn = HypergraphConv(
        in_channels=config.feature_dim,
        out_channels=config.feature_dim
    ).to(device)
    
    source_features_reshaped = source_features.permute(0, 2, 1).unsqueeze(-1)
    processed_features = hgnn(
        source_features_reshaped,
        hyperedge_matrix,
        point_hyperedge_index,
        hyperedge_centers
    )
    print(f"Processed features shape: {processed_features.shape}")
    
    # 从 target 提取特征（用于对比）
    t_zero = torch.zeros(config.batch_size, dtype=torch.long, device=device)
    target_features = unet.extract_patch_features(
        target_images, t_zero, target_images, num_patches=config.patch_size
    )
    
    # Hypergraph loss
    processed_features_flat = processed_features.squeeze(-1).permute(0, 2, 1)
    hg_loss = hypergraph_loss_fn(processed_features_flat, target_features)
    print(f"Hypergraph loss: {hg_loss.item():.4f}")
    
    # 总损失
    total_loss = diff_loss + 0.1 * hg_loss
    print(f"Total loss: {total_loss.item():.4f}")
    
    print("\n训练步骤完成！")


def example_sampling():
    """示例：采样生成图像"""
    print("\n" + "=" * 60)
    print("示例：采样生成")
    print("=" * 60)
    
    config = Config()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(config.device)
    
    # 创建模型
    unet = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        condition_channels=3
    ).to(device)
    
    diffusion = Diffusion(
        num_timesteps=100,  # 采样时可以用更少的时间步
        device=config.device
    )
    
    # 模拟 source 图像
    source_image = torch.randn(1, 3, 256, 256).to(device)
    
    # 采样
    shape = (1, 3, 256, 256)
    generated_image = diffusion.p_sample_loop(
        unet,
        shape,
        condition=source_image
    )
    
    print(f"Generated image shape: {generated_image.shape}")
    print("采样完成！")


if __name__ == "__main__":
    print("Hypergraph-guided Diffusion 使用示例\n")
    
    # 运行示例
    example_training_step()
    example_sampling()
    
    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)

