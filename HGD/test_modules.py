"""
模块测试脚本
验证所有模块可以正常工作
"""
import torch
import torch.nn.functional as F


def test_unet():
    """测试 U-Net"""
    print("=" * 50)
    print("Testing ConditionalUNet...")
    
    from models.unet import ConditionalUNet
    
    model = ConditionalUNet(
        in_channels=3,
        out_channels=3,
        model_channels=64,
        channel_mult=(1, 2, 4),
        num_res_blocks=1,
        attention_resolutions=(8,),
        image_size=64
    )
    
    B = 2
    x = torch.randn(B, 3, 64, 64)
    t = torch.randint(0, 1000, (B,))
    cond = torch.randn(B, 3, 64, 64)
    domain_id = torch.zeros(B, dtype=torch.long)
    
    noise_pred, mid_feat = model(x, t, cond, domain_id)
    
    print(f"  Input: {x.shape}")
    print(f"  Output: {noise_pred.shape}")
    print(f"  Mid features: {mid_feat.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("  ✓ U-Net test passed!")
    
    return True


def test_diffusion():
    """测试 Diffusion"""
    print("=" * 50)
    print("Testing GaussianDiffusion...")
    
    from models.unet import ConditionalUNet
    from models.diffusion import GaussianDiffusion
    
    unet = ConditionalUNet(
        in_channels=3,
        out_channels=3,
        model_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        image_size=32
    )
    
    diffusion = GaussianDiffusion(
        model=unet,
        num_timesteps=100,
        sampling_timesteps=10
    )
    
    B = 2
    x_0 = torch.randn(B, 3, 32, 32)
    cond = torch.randn(B, 3, 32, 32)
    
    # Test training forward
    loss, noise_pred, mid_feat = diffusion(x_0, cond)
    print(f"  Training loss: {loss.item():.4f}")
    print(f"  Noise pred: {noise_pred.shape}")
    
    # Test sampling (short for speed)
    print("  Testing sampling...")
    with torch.no_grad():
        sample = diffusion.sample(cond[:1], use_ddim=True)
    print(f"  Sample: {sample.shape}")
    print("  ✓ Diffusion test passed!")
    
    return True


def test_hypergraph_construction():
    """测试 Hypergraph 构建"""
    print("=" * 50)
    print("Testing HypergraphConstructor...")
    
    from hypergraph.construction import HypergraphConstructor, soft_kmeans_clustering
    
    # Test soft k-means
    x = torch.randn(2, 100, 64)
    memberships, centers = soft_kmeans_clustering(x, num_clusters=9)
    print(f"  Soft k-means - Input: {x.shape}, Memberships: {memberships.shape}, Centers: {centers.shape}")
    
    # Test constructor
    constructor = HypergraphConstructor(num_clusters=9)
    features = torch.randn(2, 256, 8, 8)
    
    patch_feat, memberships, hyperedge_feat, incidence = constructor(features, return_binary=True)
    print(f"  Constructor - Input: {features.shape}")
    print(f"    Patch features: {patch_feat.shape}")
    print(f"    Memberships: {memberships.shape}")
    print(f"    Hyperedge features: {hyperedge_feat.shape}")
    print(f"    Incidence matrix: {incidence.shape}")
    
    # Verify membership sums to ~1
    membership_sum = memberships.sum(dim=-1).mean()
    print(f"    Avg membership sum: {membership_sum.item():.4f} (should be ~1.0)")
    print("  ✓ Hypergraph construction test passed!")
    
    return True


def test_hgnn():
    """测试 HGNN"""
    print("=" * 50)
    print("Testing HGNN...")
    
    from hypergraph.hgnn import HypergraphConv, HypergraphEncoder, DualHypergraphEncoder
    
    B, N, D, K = 2, 64, 128, 9
    node_feat = torch.randn(B, N, D)
    memberships = F.softmax(torch.randn(B, N, K), dim=-1)
    centers = torch.randn(B, K, D)
    
    # Test HypergraphConv
    conv = HypergraphConv(D, 256)
    out = conv(node_feat, memberships, centers)
    print(f"  HypergraphConv - Input: {node_feat.shape}, Output: {out.shape}")
    
    # Test HypergraphEncoder
    encoder = HypergraphEncoder(D, 256, 128, num_layers=2)
    out = encoder(node_feat, memberships, centers)
    print(f"  HypergraphEncoder - Output: {out.shape}, Norm: {out.norm(dim=-1).mean():.4f}")
    
    # Test DualHypergraphEncoder
    dual_encoder = DualHypergraphEncoder(D, 256, 128)
    source_feat = torch.randn(B, N, D)
    target_feat = torch.randn(B, N, D)
    src_enc, tgt_enc = dual_encoder(source_feat, target_feat, memberships, centers)
    print(f"  DualHypergraphEncoder - Source: {src_enc.shape}, Target: {tgt_enc.shape}")
    print("  ✓ HGNN test passed!")
    
    return True


def test_losses():
    """测试损失函数"""
    print("=" * 50)
    print("Testing Losses...")
    
    from losses.diffusion_loss import DiffusionLoss
    from losses.hypergraph_loss import (
        InfoNCELoss, 
        WeightedInfoNCELoss, 
        HypergraphContrastiveLoss,
        HypergraphStructureLoss
    )
    
    # Diffusion loss
    diff_loss = DiffusionLoss()
    pred = torch.randn(4, 3, 32, 32)
    target = torch.randn(4, 3, 32, 32)
    loss = diff_loss(pred, target)
    print(f"  DiffusionLoss: {loss.item():.4f}")
    
    # InfoNCE loss
    info_nce = InfoNCELoss()
    feat_a = torch.randn(64, 128)
    feat_b = torch.randn(64, 128)
    loss = info_nce(feat_a, feat_b)
    print(f"  InfoNCELoss: {loss.item():.4f}")
    
    # Weighted InfoNCE
    weighted_nce = WeightedInfoNCELoss()
    loss = weighted_nce(feat_a, feat_b)
    print(f"  WeightedInfoNCELoss: {loss.item():.4f}")
    
    # HypergraphContrastiveLoss
    hg_loss = HypergraphContrastiveLoss(
        feature_dim=256,
        hidden_dim=128,
        output_dim=128,
        num_clusters=9
    )
    source_feat = torch.randn(2, 256, 8, 8)
    target_feat = torch.randn(2, 256, 8, 8)
    loss, info = hg_loss(source_feat, target_feat)
    print(f"  HypergraphContrastiveLoss: {loss.item():.4f}")
    print(f"    Info: {info}")
    
    # HypergraphStructureLoss
    struct_loss = HypergraphStructureLoss(num_clusters=9)
    loss, info = struct_loss(source_feat, target_feat)
    print(f"  HypergraphStructureLoss: {loss.item():.4f}")
    print("  ✓ Losses test passed!")
    
    return True


def test_integration():
    """集成测试: 完整的训练 step"""
    print("=" * 50)
    print("Testing Integration (full training step)...")
    
    from models.unet import ConditionalUNet
    from models.diffusion import GaussianDiffusion
    from losses.hypergraph_loss import HypergraphContrastiveLoss
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # 创建模型
    unet = ConditionalUNet(
        in_channels=3,
        out_channels=3,
        model_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        image_size=32,
        return_mid_features=True
    ).to(device)
    
    diffusion = GaussianDiffusion(
        model=unet,
        num_timesteps=100,
        sampling_timesteps=10
    ).to(device)
    
    # 获取 mid channels
    mid_channels = unet.mid_channels
    
    hg_loss_fn = HypergraphContrastiveLoss(
        feature_dim=mid_channels,
        hidden_dim=64,
        output_dim=64,
        num_clusters=4
    ).to(device)
    
    # 模拟训练 step
    optimizer = torch.optim.Adam(list(unet.parameters()) + list(hg_loss_fn.parameters()), lr=1e-4)
    
    B = 2
    source = torch.randn(B, 3, 32, 32).to(device)
    target = torch.randn(B, 3, 32, 32).to(device)
    
    # Forward
    diff_loss, noise_pred, mid_feat_target = diffusion(target, source)
    
    # 获取 source 的 mid features
    with torch.no_grad():
        t_dummy = torch.zeros(B, dtype=torch.long, device=device)
        _, mid_feat_source = unet(source, t_dummy, source)
    
    # Hypergraph loss
    hg_loss, hg_info = hg_loss_fn(mid_feat_source.detach(), mid_feat_target)
    
    # Total loss
    total_loss = diff_loss + 0.1 * hg_loss
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"  Diffusion loss: {diff_loss.item():.4f}")
    print(f"  Hypergraph loss: {hg_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
    print("  ✓ Integration test passed!")
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("HGD Module Tests")
    print("=" * 50)
    
    tests = [
        ("U-Net", test_unet),
        ("Diffusion", test_diffusion),
        ("Hypergraph Construction", test_hypergraph_construction),
        ("HGNN", test_hgnn),
        ("Losses", test_losses),
        ("Integration", test_integration),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"  ✗ {name} test failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to train.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()

