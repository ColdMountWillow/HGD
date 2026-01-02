"""
基础测试脚本：验证各个模块是否正常工作
"""
import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models import UNet, Diffusion
from hypergraph import construct_hyperedges
from hypergraph.hgnn import HypergraphConv
from losses import DiffusionLoss, HypergraphContrastiveLoss


def test_unet():
    """测试 U-Net"""
    print("Testing UNet...")
    config = Config()
    device = torch.device("cpu")
    
    unet = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,  # 使用较小的通道数以加快测试
        channel_multipliers=(1, 2, 4),
        num_res_blocks=1,
        attention_resolutions=(8,),
        condition_channels=3
    ).to(device)
    
    # 测试 forward
    x = torch.randn(2, 3, 64, 64).to(device)
    t = torch.randint(0, 100, (2,)).long().to(device)
    condition = torch.randn(2, 3, 64, 64).to(device)
    
    output = unet(x, t, condition)
    assert output.shape == (2, 3, 64, 64), f"Expected (2, 3, 64, 64), got {output.shape}"
    print("  ✓ UNet forward pass: OK")
    
    # 测试特征提取
    features = unet.extract_patch_features(x, t, condition, num_patches=32)
    assert features.shape[0] == 2 and features.shape[1] == 32, f"Expected (2, 32, ...), got {features.shape}"
    print("  ✓ UNet feature extraction: OK")
    print("UNet test passed!\n")


def test_diffusion():
    """测试 Diffusion"""
    print("Testing Diffusion...")
    device = torch.device("cpu")
    
    diffusion = Diffusion(
        num_timesteps=100,
        device=device
    )
    
    # 测试 q_sample
    x0 = torch.randn(2, 3, 64, 64).to(device)
    t = torch.randint(0, 100, (2,)).long().to(device)
    noise = torch.randn_like(x0)
    
    xt = diffusion.q_sample(x0, t, noise)
    assert xt.shape == x0.shape, f"Shape mismatch: {xt.shape} vs {x0.shape}"
    print("  ✓ Diffusion q_sample: OK")
    
    # 测试 p_sample（需要模型）
    class DummyModel(torch.nn.Module):
        def forward(self, x, t, condition=None):
            return torch.randn_like(x)
    
    model = DummyModel()
    x_prev = diffusion.p_sample(model, xt, t, condition=x0)
    assert x_prev.shape == xt.shape, f"Shape mismatch: {x_prev.shape} vs {xt.shape}"
    print("  ✓ Diffusion p_sample: OK")
    print("Diffusion test passed!\n")


def test_hypergraph_construction():
    """测试 Hypergraph 构建"""
    print("Testing Hypergraph Construction...")
    device = torch.device("cpu")
    
    batch_size = 2
    num_patches = 32
    feature_dim = 128
    
    patch_features = torch.randn(batch_size, num_patches, feature_dim).to(device)
    
    hyperedge_matrix, point_hyperedge_index, hyperedge_centers = construct_hyperedges(
        patch_features,
        num_clusters=5,
        threshold=0.15
    )
    
    assert hyperedge_matrix.shape[0] == batch_size
    assert hyperedge_matrix.shape[1] == 5
    assert point_hyperedge_index.shape[0] == batch_size
    assert hyperedge_centers.shape[0] == batch_size
    assert hyperedge_centers.shape[1] == feature_dim
    assert hyperedge_centers.shape[2] == 5
    
    print("  ✓ Hypergraph construction: OK")
    print("Hypergraph construction test passed!\n")


def test_hypergraph_conv():
    """测试 Hypergraph Convolution"""
    print("Testing HypergraphConv...")
    device = torch.device("cpu")
    
    batch_size = 2
    num_patches = 32
    feature_dim = 128
    
    hgnn = HypergraphConv(
        in_channels=feature_dim,
        out_channels=feature_dim
    ).to(device)
    
    node_features = torch.randn(batch_size, feature_dim, num_patches, 1).to(device)
    hyperedge_matrix = torch.randint(0, num_patches, (batch_size, 5, num_patches)).long().to(device)
    point_hyperedge_index = torch.randint(0, 5, (batch_size, num_patches, 3)).long().to(device)
    hyperedge_centers = torch.randn(batch_size, feature_dim, 5).to(device)
    
    output = hgnn(node_features, hyperedge_matrix, point_hyperedge_index, hyperedge_centers)
    
    assert output.shape[0] == batch_size
    assert output.shape[1] == feature_dim
    assert output.shape[2] == num_patches
    assert output.shape[3] == 1
    
    print("  ✓ HypergraphConv forward: OK")
    print("HypergraphConv test passed!\n")


def test_losses():
    """测试损失函数"""
    print("Testing Loss Functions...")
    device = torch.device("cpu")
    
    # Diffusion loss
    diff_loss_fn = DiffusionLoss()
    pred_noise = torch.randn(2, 3, 64, 64).to(device)
    true_noise = torch.randn(2, 3, 64, 64).to(device)
    
    diff_loss = diff_loss_fn(pred_noise, true_noise)
    assert diff_loss.item() >= 0, "Loss should be non-negative"
    print("  ✓ Diffusion loss: OK")
    
    # Hypergraph loss
    hg_loss_fn = HypergraphContrastiveLoss()
    source_features = torch.randn(2, 32, 128).to(device)
    target_features = torch.randn(2, 32, 128).to(device)
    
    hg_loss = hg_loss_fn(source_features, target_features)
    assert hg_loss.item() >= 0, "Loss should be non-negative"
    print("  ✓ Hypergraph loss: OK")
    print("Loss functions test passed!\n")


def test_end_to_end():
    """端到端测试"""
    print("Testing End-to-End Pipeline...")
    device = torch.device("cpu")
    
    config = Config()
    config.batch_size = 2
    config.image_size = 64
    config.num_timesteps = 50
    config.patch_size = 16
    config.num_hyperedges = 5
    config.feature_dim = 128
    
    # 创建模型
    unet = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        condition_channels=3
    ).to(device)
    
    diffusion = Diffusion(num_timesteps=config.num_timesteps, device=device)
    hgnn = HypergraphConv(in_channels=config.feature_dim, out_channels=config.feature_dim).to(device)
    diff_loss_fn = DiffusionLoss()
    hg_loss_fn = HypergraphContrastiveLoss()
    
    # 模拟数据
    source = torch.randn(config.batch_size, 3, config.image_size, config.image_size).to(device)
    target = torch.randn(config.batch_size, 3, config.image_size, config.image_size).to(device)
    t = torch.randint(0, config.num_timesteps, (config.batch_size,)).long().to(device)
    noise = torch.randn_like(target)
    
    # Forward diffusion
    x_t = diffusion.q_sample(target, t, noise)
    
    # Predict noise
    pred_noise = unet(x_t, t, source)
    diff_loss = diff_loss_fn(pred_noise, noise)
    
    # Extract features
    source_features = unet.extract_patch_features(x_t, t, source, num_patches=config.patch_size)
    
    # Build hypergraph
    hyperedge_matrix, point_hyperedge_index, hyperedge_centers = construct_hyperedges(
        source_features, num_clusters=config.num_hyperedges
    )
    
    # Hypergraph conv
    source_features_reshaped = source_features.permute(0, 2, 1).unsqueeze(-1)
    processed_features = hgnn(source_features_reshaped, hyperedge_matrix, point_hyperedge_index, hyperedge_centers)
    processed_features_flat = processed_features.squeeze(-1).permute(0, 2, 1)
    
    # Target features
    t_zero = torch.zeros(config.batch_size, dtype=torch.long, device=device)
    target_features = unet.extract_patch_features(target, t_zero, target, num_patches=config.patch_size)
    
    # Hypergraph loss
    hg_loss = hg_loss_fn(processed_features_flat, target_features)
    
    # Total loss
    total_loss = diff_loss + 0.1 * hg_loss
    
    print(f"  Diffusion loss: {diff_loss.item():.4f}")
    print(f"  Hypergraph loss: {hg_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
    print("  ✓ End-to-end pipeline: OK")
    print("End-to-end test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Basic Tests")
    print("=" * 60 + "\n")
    
    try:
        test_unet()
        test_diffusion()
        test_hypergraph_construction()
        test_hypergraph_conv()
        test_losses()
        test_end_to_end()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

