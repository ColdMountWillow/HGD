# Hypergraph-guided Diffusion for Unpaired Virtual Stain Translation

## 方法概述

本项目实现了结合 **Diffusion-based Image-to-Image Translation** 与 **Patch-level Hypergraph Learning** 的无配对虚拟染色迁移系统。

### 核心思想

1. **Conditional Diffusion Model**: 使用 U-Net backbone 的条件扩散模型，以 source stain 图像为条件生成 target stain 图像
2. **Hypergraph Structure Constraint**: 从 U-Net 中间特征层提取 patch features，构建 hypergraph 并约束跨 domain 的结构一致性
3. **Unpaired Training**: 无需像素级配对数据，仅需两个 domain 的图像集合

### 系统架构

```
Source Image (Domain A)
    ↓
[U-Net Encoder] → [Feature Extraction] → [Patch Features]
    ↓                                        ↓
[U-Net Decoder]                    [Hypergraph Construction]
    ↓                                        ↓
Noisy Target Image                  [Hypergraph Contrastive Loss]
    ↓
Target Image (Domain B)
```

### 损失函数

- **Diffusion Noise Prediction Loss**: 标准 DDPM/DDIM 损失
- **Hypergraph Contrastive Loss**: 约束 source 和 target domain 的 patch-level 结构一致性

## 项目结构

```
HypergraphDiffusion/
├── models/
│   ├── __init__.py
│   ├── unet.py              # Diffusion U-Net backbone
│   └── diffusion.py         # Diffusion 调度器
├── hypergraph/
│   ├── __init__.py
│   ├── construction.py      # Hypergraph 构建
│   └── hgnn.py             # Hypergraph Neural Network
├── losses/
│   ├── __init__.py
│   ├── diffusion_loss.py   # Diffusion 损失
│   └── hypergraph_loss.py  # Hypergraph 对比损失
├── data/
│   └── dataset.py          # 数据加载器
├── train.py                # 训练脚本
├── config.py               # 配置文件
└── README.md
```

## 使用方法

### 训练

```bash
python train.py --config configs/default.yaml
```

### 配置参数

主要参数：

- `num_timesteps`: Diffusion 时间步数 (默认: 1000)
- `num_hyperedges`: Hyperedge 数量 (默认: 9)
- `patch_size`: Patch 采样大小 (默认: 64)
- `hypergraph_loss_weight`: Hypergraph 损失权重 (默认: 0.1)

## 实验设计

- **Baseline**: 无 hypergraph 的 diffusion
- **Ablation**: 不同 hyperedge 数量 / patch 尺度
- **Visualization**: 结构保持 vs 纹理变化的定性分析
