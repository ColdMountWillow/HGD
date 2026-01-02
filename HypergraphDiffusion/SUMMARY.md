# Hypergraph-guided Diffusion 系统总结

## 📋 项目概述

本项目实现了 **Hypergraph-guided Diffusion for Unpaired Virtual Stain Translation**，将 diffusion-based image-to-image translation 与 patch-level hypergraph learning 结合，用于无配对病理虚拟染色迁移。

## 🎯 核心创新点

1. **Conditional Diffusion + Hypergraph Constraint**: 
   - 使用条件扩散模型进行图像到图像转换
   - 通过 hypergraph 约束跨 domain 的结构一致性

2. **Patch-level Hypergraph Learning**:
   - 从 U-Net 中间特征层提取 patch features
   - 使用 soft k-means 构建 hyperedges
   - Hypergraph Neural Network 进行信息传播

3. **Unpaired Training**:
   - 无需像素级配对数据
   - 通过对比学习约束结构一致性
   - 不使用 cycle consistency

## 📁 项目结构

```
HypergraphDiffusion/
├── models/              # 模型定义
│   ├── unet.py         # Conditional U-Net
│   └── diffusion.py    # Diffusion 调度器
├── hypergraph/         # Hypergraph 模块
│   ├── construction.py # Hypergraph 构建
│   └── hgnn.py        # Hypergraph Neural Network
├── losses/            # 损失函数
│   ├── diffusion_loss.py
│   └── hypergraph_loss.py
├── data/              # 数据加载
│   └── dataset.py
├── train.py           # 训练脚本
├── config.py          # 配置文件
├── example_usage.py   # 使用示例
├── ARCHITECTURE.md    # 架构说明
└── README.md          # 项目说明
```

## 🔧 关键模块说明

### 1. UNet (`models/unet.py`)

**功能**: Conditional Diffusion U-Net

**关键方法**:
- `forward(x, timestep, condition)`: 预测噪声
- `extract_patch_features(...)`: 提取 patch features 用于 hypergraph

**输入/输出**:
- 输入: 噪声图像 [B, C, H, W], 时间步 [B], 条件图像 [B, C, H, W]
- 输出: 预测噪声 [B, C, H, W]

### 2. Hypergraph Construction (`hypergraph/construction.py`)

**功能**: 从 patch features 构建 hypergraph

**关键函数**:
- `soft_k_means(...)`: Soft k-means 聚类
- `construct_hyperedges(...)`: 构建 hyperedges

**输入/输出**:
- 输入: Patch features [B, num_patches, feature_dim]
- 输出: Hyperedge matrix, Point-hyperedge index, Hyperedge centers

### 3. HypergraphConv (`hypergraph/hgnn.py`)

**功能**: Hypergraph 信息传播

**流程**:
1. Node → Hyperedge: 聚合每个 hyperedge 内的 nodes
2. Hyperedge → Node: 聚合每个 node 所属的 hyperedges

### 4. Loss Functions

**Diffusion Loss**: 标准 DDPM noise prediction loss

**Hypergraph Loss**: InfoNCE 对比损失，约束 source 和 target domain 的 patch features 一致性

## 🚀 使用方法

### 训练

```bash
# 准备数据
# data/
#   ├── H&E/    # Source domain
#   └── PAS/    # Target domain

# 训练
python train.py \
    --data_root ./data \
    --batch_size 4 \
    --num_epochs 100 \
    --learning_rate 1e-4
```

### 配置参数

主要参数（在 `config.py` 中）:
- `num_timesteps`: Diffusion 时间步数 (默认: 1000)
- `num_hyperedges`: Hyperedge 数量 (默认: 9)
- `patch_size`: Patch 采样数量 (默认: 64)
- `hypergraph_loss_weight`: Hypergraph 损失权重 (默认: 0.1)
- `temperature`: 对比学习温度 (默认: 0.07)

### 运行示例

```bash
python example_usage.py
```

## 📊 训练流程

1. **Forward Diffusion**: 对 target 图像添加噪声
2. **Noise Prediction**: U-Net 预测噪声（以 source 为条件）
3. **Feature Extraction**: 从 U-Net bottleneck 提取 patch features
4. **Hypergraph Construction**: 构建 hyperedges
5. **Hypergraph Convolution**: 信息传播
6. **Loss Computation**: 
   - Diffusion loss: MSE(predicted_noise, true_noise)
   - Hypergraph loss: InfoNCE(source_features, target_features)
7. **Backward & Update**: 反向传播更新参数

## 🧪 实验设计建议

### Baseline 对比
- **Baseline 1**: 无 hypergraph 的 diffusion（`hypergraph_loss_weight=0`）
- **Baseline 2**: 标准 CycleGAN（如果可用）

### Ablation Studies
1. **Hyperedge 数量**: 3, 6, 9, 12, 15
2. **Patch 数量**: 32, 64, 128, 256
3. **Hypergraph Loss Weight**: 0.01, 0.05, 0.1, 0.2, 0.5
4. **Temperature**: 0.05, 0.07, 0.1, 0.15

### 评估指标
- **结构保持**: SSIM, LPIPS
- **风格迁移**: FID, KID
- **定性分析**: 可视化对比

## 🔍 代码关键点

### 1. Patch Feature 提取位置

在 `UNet.extract_patch_features()` 中，从 **bottleneck** 层提取特征：
- 位置：U-Net 的中间层（encoder 和 decoder 之间）
- 原因：bottleneck 包含最丰富的语义信息

### 2. Hypergraph 构建时机

在训练时，对 **加噪后的图像** 提取 features：
- 使用 `x_t`（加噪图像）而不是 `x_0`（原始图像）
- 这样可以学习在去噪过程中保持结构一致性

### 3. 对比损失设计

使用 **InfoNCE** 损失：
- Positive pairs: 对应位置的 patches
- Negative pairs: 所有其他 patches
- 温度参数控制 softness

## ⚠️ 注意事项

1. **内存占用**: 
   - Hypergraph 构建和对比损失可能占用较多内存
   - 建议 batch_size ≤ 4（单卡 GPU）

2. **训练稳定性**:
   - 建议使用梯度裁剪
   - 可以先用较小的 `hypergraph_loss_weight` 开始训练

3. **数据准备**:
   - 确保 source 和 target domain 的图像数量足够
   - 图像尺寸建议 256x256 或 512x512

## 🔮 后续扩展方向

1. **Multi-scale Hypergraph**: 在不同分辨率层级构建多个 hypergraph
2. **Attention Conditioning**: 在 cross-attention 中融入 hypergraph 信息
3. **Adaptive Hyperedge Number**: 根据图像内容自适应调整
4. **Object-level Hypergraph**: 使用 object detector 提取更高层次特征

## 📝 引用格式（建议）

如果用于论文，建议引用：

```bibtex
@article{hypergraph_diffusion_2024,
  title={Hypergraph-guided Diffusion for Unpaired Virtual Stain Translation},
  author={Your Name},
  journal={MICCAI / Medical Image Analysis},
  year={2024}
}
```

## 📧 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

---

**最后更新**: 2024年

