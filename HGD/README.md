# Hypergraph-guided Diffusion for Unpaired Virtual Stain Translation (HGD)

## 1. 方法概述

**Hypergraph-guided Diffusion (HGD)** 是一种用于无配对病理虚拟染色迁移的新框架。

### 核心思想

1. **Conditional Diffusion** 作为基础生成模型，以 source stain 图像为条件生成 target stain 图像
2. **Patch-level Hypergraph Learning** 从 U-Net 中间特征提取 patch tokens，通过 soft k-means 聚类构建 hyperedges
3. **Hypergraph Contrastive Loss** 约束跨 domain 的 patch-level 结构拓扑一致，允许风格变化但保持组织结构

### 关键创新

- 不依赖 GT segmentation 或 cycle consistency
- 通过学习到的 patch-level hypergraph 结构约束实现结构保持
- 支持 unpaired 训练范式

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Hypergraph-guided Diffusion (HGD)                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [Source Image A] ─────────────────────────────────────────┐                 │
│         │                                                   │                 │
│         ▼                                                   │                 │
│  ┌──────────────────┐                                      │                 │
│  │  Conditional     │                                      │                 │
│  │  Diffusion UNet  │  ◄──── [Noisy Target + Source Cond]  │                 │
│  │  ┌────────────┐  │                                      │                 │
│  │  │ Mid-Block  │──┼──► [Patch Features F_t]              │                 │
│  │  └────────────┘  │           │                          │                 │
│  └──────────────────┘           │                          ▼                 │
│         │                       │        [Source Mid Features F_s]           │
│         ▼                       │                          │                 │
│  [Noise Prediction]             │                          │                 │
│         │                       ▼                          ▼                 │
│         │            ┌─────────────────────────────────────────┐             │
│         │            │        Hypergraph Module                │             │
│         │            │  ┌────────────────────────────────────┐ │             │
│         │            │  │ 1. Flatten to Patch Tokens (B,N,D) │ │             │
│         │            │  │ 2. Soft K-Means Clustering         │ │             │
│         │            │  │ 3. Build Hyperedges (membership)   │ │             │
│         │            │  │ 4. HGNN Message Passing            │ │             │
│         │            │  │ 5. Contrastive Loss                │ │             │
│         │            │  └────────────────────────────────────┘ │             │
│         │            └─────────────────────────────────────────┘             │
│         │                              │                                     │
│         ▼                              ▼                                     │
│  L_diffusion                    L_hypergraph                                 │
│         │                              │                                     │
│         └──────────────┬───────────────┘                                     │
│                        ▼                                                     │
│            Total Loss = L_diff + λ * L_hg                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 项目结构

```
HGD/
├── models/
│   ├── __init__.py
│   ├── unet.py              # Conditional Diffusion U-Net
│   ├── diffusion.py         # DDPM/DDIM Diffusion 核心
│   └── condition_encoder.py # 条件编码器 (可选)
├── hypergraph/
│   ├── __init__.py
│   ├── construction.py      # Hypergraph 构建 (soft k-means)
│   └── hgnn.py              # Hypergraph Neural Network
├── losses/
│   ├── __init__.py
│   ├── diffusion_loss.py    # Noise prediction loss
│   └── hypergraph_loss.py   # Hypergraph contrastive loss
├── data/
│   ├── __init__.py
│   └── unpaired_dataset.py  # Unpaired 数据加载器
├── train.py                 # 训练脚本
├── sample.py                # 采样/推理脚本
├── test_modules.py          # 模块测试脚本
├── config.py                # 配置文件
└── requirements.txt         # 依赖
```

---

## 4. 快速开始

### 安装依赖

```bash
cd HGD
pip install -r requirements.txt
```

### 数据准备

```
data/
├── HE/       # Source domain (e.g., H&E staining)
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
└── IHC/      # Target domain (e.g., IHC staining)
    ├── img_001.png
    ├── img_002.png
    └── ...
```

**注意**: Unpaired 模式下，两个 domain 的图像不需要一一对应。

### 测试模块

```bash
python test_modules.py
```

### 训练

```bash
# Baseline: 纯 conditional diffusion
python train.py --data_root ./data --domain_a HE --domain_b IHC

# 完整 HGD: 带 hypergraph loss
python train.py --data_root ./data --domain_a HE --domain_b IHC \
    --use_hypergraph --lambda_hg 0.1

# 自定义配置
python train.py --data_root ./data \
    --image_size 256 \
    --model_channels 128 \
    --batch_size 4 \
    --num_epochs 100 \
    --use_hypergraph \
    --lambda_hg 0.1 \
    --num_clusters 9
```

### 推理/采样

```bash
# 处理单张图像
python sample.py --checkpoint ./checkpoints/best.pt \
    --input_image ./test.png \
    --output_dir ./results

# 处理整个目录
python sample.py --checkpoint ./checkpoints/best.pt \
    --input_dir ./test_images \
    --output_dir ./results

# 交互模式
python sample.py --checkpoint ./checkpoints/best.pt --interactive
```

---

## 5. 核心模块详解

### 5.1 ConditionalUNet (`models/unet.py`)

```python
# 输入
x: (B, C, H, W)      # noisy image
t: (B,)              # timesteps
cond: (B, C, H, W)   # condition image (source stain)
domain_id: (B,)      # optional domain embedding

# 输出
noise_pred: (B, C, H, W)                    # 预测的噪声
mid_features: (B, C_mid, H_mid, W_mid)      # 中间特征 (用于 hypergraph)
```

### 5.2 HypergraphConstructor (`hypergraph/construction.py`)

```python
# 输入
features: (B, C, H, W)  # U-Net 中间特征

# 处理流程
1. Flatten: (B, C, H, W) -> (B, N, C) where N = H*W
2. Soft K-Means: 计算 soft membership matrix
3. 输出 hyperedge 结构

# 输出
patch_features: (B, N, D)        # N 个 patch 的特征
memberships: (B, N, K)           # soft membership 矩阵
hyperedge_features: (B, K, D)    # K 个 hyperedge 的特征 (簇中心)
```

### 5.3 HypergraphContrastiveLoss (`losses/hypergraph_loss.py`)

```python
# 输入
source_features: (B, C, H, W)  # source domain 中间特征
target_features: (B, C, H, W)  # target domain 中间特征

# 处理流程
1. 从 target features 构建 hypergraph
2. 使用 DualHypergraphEncoder 分别编码 source/target
3. Node-wise bidirectional InfoNCE loss

# 输出
loss: scalar                    # contrastive loss
info: dict                      # 调试信息
```

---

## 6. 训练流程伪代码

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        source = batch['source']  # (B, C, H, W) from domain A
        target = batch['target']  # (B, C, H, W) from domain B

        # 1. Diffusion forward
        t = random_timesteps(B)
        noise = randn_like(target)
        x_t = q_sample(target, t, noise)  # 加噪

        # 2. Noise prediction
        noise_pred, mid_feat_target = unet(x_t, t, cond=source)
        loss_diff = mse_loss(noise_pred, noise)

        # 3. Hypergraph loss (optional)
        if use_hypergraph:
            # 获取 source 的中间特征
            with no_grad():
                _, mid_feat_source = unet(source, 0, source)

            # 构建 hypergraph 并计算 contrastive loss
            loss_hg = hypergraph_loss(mid_feat_source, mid_feat_target)

        # 4. Total loss
        loss = loss_diff + lambda_hg * loss_hg

        # 5. Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 7. 实验设计

### 7.1 消融实验

| 实验       | Hypergraph | λ_hg | 说明                     |
| ---------- | ---------- | ---- | ------------------------ |
| Baseline   | ✗          | -    | 纯 conditional diffusion |
| +HG (0.05) | ✓          | 0.05 | 轻量 hypergraph 约束     |
| +HG (0.1)  | ✓          | 0.1  | 标准 hypergraph 约束     |
| +HG (0.2)  | ✓          | 0.2  | 强 hypergraph 约束       |

### 7.2 Hyperedge 数量消融

| K   | 描述       |
| --- | ---------- |
| 4   | 粗粒度结构 |
| 9   | 默认配置   |
| 16  | 细粒度结构 |
| 32  | 精细结构   |

### 7.3 评估指标

- **FID**: Fréchet Inception Distance (图像质量)
- **SSIM**: 结构相似度 (如有 paired test data)
- **Downstream Task**: 分割/分类性能

---

## 8. 可扩展方向

1. **Object-level Hypergraph**: 使用预训练检测器提取对象级节点
2. **Cross-attention Conditioning**: 替代 concat conditioning
3. **Multi-scale Hypergraph**: 从多个 U-Net 层提取特征
4. **Classifier-free Guidance**: 增强条件控制
5. **Latent Diffusion**: 在潜空间进行扩散 (更高效)

---

## 9. 常见问题

**Q: 训练需要多长时间？**
A: 在 RTX 3090 上，256×256 图像，batch_size=4，约 100 epochs 需要 ~20 小时。

**Q: 如何调整显存占用？**
A: 减小 `model_channels`、`batch_size` 或 `image_size`。

**Q: 生成结果模糊？**
A: 尝试增加 `sampling_steps` 或调整 `ddim_eta`。

---

## 10. Citation

如果您使用此代码，请引用:

```bibtex
@article{hgd2025,
  title={Hypergraph-guided Diffusion for Unpaired Virtual Stain Translation},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

参考工作:

- Graph Conditioned Diffusion (Cechnicka et al., 2025)
- STNHCL: Patch-Wise Hypergraph Contrastive Learning (Wei et al., 2025)
