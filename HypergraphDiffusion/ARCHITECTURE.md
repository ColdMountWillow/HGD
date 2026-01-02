# 系统架构说明

## 1. 方法整体说明

本系统实现了 **Hypergraph-guided Diffusion for Unpaired Virtual Stain Translation**，核心思想是：

1. **Conditional Diffusion Model**: 使用 U-Net 作为 backbone 的条件扩散模型，以 source stain 图像为条件，逐步去噪生成 target stain 图像。

2. **Hypergraph Structure Constraint**: 从 U-Net 的中间特征层（bottleneck）提取 patch-level features，通过 soft k-means 聚类构建 hyperedges，使用 Hypergraph Neural Network (HGNN) 进行信息传播，约束跨 domain 的结构一致性。

3. **Unpaired Training**: 无需像素级配对数据，只需要两个 domain 的图像集合，通过对比学习约束结构一致性。

## 2. 系统架构图（模块级）

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
└─────────────────────────────────────────────────────────────┘

Input: Source Image (Domain A)  [B, 3, H, W]
       Target Image (Domain B)  [B, 3, H, W]  (unpaired)

┌─────────────────────────────────────────────────────────────┐
│  Step 1: Forward Diffusion                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  q(x_t | x_0) = √(α̅_t) x_0 + √(1-α̅_t) ε            │  │
│  │  t ~ Uniform(0, T)                                   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: U-Net Forward Pass                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  UNet(x_t, t, condition=source_image)                │  │
│  │  ├── Encoder: Down-sampling                            │  │
│  │  ├── Bottleneck: Feature Extraction ←───┐             │  │
│  │  └── Decoder: Up-sampling              │             │  │
│  └─────────────────────────────────────────┼─────────────┘  │
│                                              │                │
│  ┌───────────────────────────────────────────┘                │
│  │  Extract Patch Features                                  │
│  │  [B, num_patches, feature_dim]                          │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Hypergraph Construction                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Soft K-Means Clustering                               │  │
│  │  ├── Memberships: [B, num_patches, num_hyperedges]    │  │
│  │  ├── Centers: [B, feature_dim, num_hyperedges]       │  │
│  │  └── Hyperedge Matrix & Point-Hyperedge Index         │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  HypergraphConv                                        │  │
│  │  ├── Node → Hyperedge Aggregation                     │  │
│  │  └── Hyperedge → Node Aggregation                    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Loss Computation                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  L_total = λ_diff * L_diff + λ_hg * L_hg             │  │
│  │                                                         │  │
│  │  L_diff = MSE(predicted_noise, true_noise)            │  │
│  │                                                         │  │
│  │  L_hg = InfoNCE(source_features, target_features)   │  │
│  │         (contrastive loss on hypergraph features)     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 3. 关键模块说明

### 3.1 Diffusion U-Net (`models/unet.py`)

**输入**:
- `x`: 噪声图像 [B, C, H, W]
- `timestep`: 时间步 [B]
- `condition`: 条件图像（source）[B, C, H, W]

**输出**:
- `predicted_noise`: 预测的噪声 [B, C, H, W]

**特征提取**:
- `extract_patch_features()`: 从 bottleneck 提取 patch features [B, num_patches, feature_dim]

**架构**:
- Encoder: 下采样 + Residual Blocks + Attention
- Bottleneck: 特征提取点（用于 hypergraph）
- Decoder: 上采样 + Residual Blocks + Attention

### 3.2 Hypergraph Construction (`hypergraph/construction.py`)

**输入**:
- `patch_features`: [B, num_patches, feature_dim]

**输出**:
- `hyperedge_matrix`: [B, num_hyperedges, num_patches] - 每个 hyperedge 包含的 points
- `point_hyperedge_index`: [B, num_patches, max_edges_per_point] - 每个 point 属于的 hyperedges
- `hyperedge_features`: [B, feature_dim, num_hyperedges] - 聚类中心

**方法**:
- Soft k-means 聚类（temperature=0.2）
- 基于隶属度阈值构建 hyperedges

### 3.3 Hypergraph Neural Network (`hypergraph/hgnn.py`)

**输入**:
- `node_features`: [B, C, N, 1]
- `hyperedge_matrix`: [B, K, N]
- `point_hyperedge_index`: [B, N, M]
- `hyperedge_centers`: [B, C, K]

**输出**:
- `updated_node_features`: [B, out_channels, N, 1]

**流程**:
1. Node → Hyperedge: 聚合每个 hyperedge 内的 nodes
2. Hyperedge → Node: 聚合每个 node 所属的 hyperedges

### 3.4 Loss Functions

**Diffusion Loss** (`losses/diffusion_loss.py`):
- MSE(predicted_noise, true_noise)

**Hypergraph Contrastive Loss** (`losses/hypergraph_loss.py`):
- InfoNCE loss 约束 source 和 target domain 的 patch features 一致性
- 使用 L2 归一化 + 温度缩放

## 4. 训练流程伪代码

```python
for epoch in range(num_epochs):
    for batch (source_images, target_images) in dataloader:
        # 1. Forward Diffusion
        t ~ Uniform(0, T)
        noise ~ N(0, I)
        x_t = q_sample(target_images, t, noise)
        
        # 2. Predict Noise
        predicted_noise = UNet(x_t, t, condition=source_images)
        
        # 3. Diffusion Loss
        L_diff = MSE(predicted_noise, noise)
        
        # 4. Extract Patch Features
        source_features = UNet.extract_patch_features(x_t, t, source_images)
        target_features = UNet.extract_patch_features(target_images, 0, target_images)
        
        # 5. Build Hypergraph
        hyperedge_matrix, point_hyperedge_index, centers = construct_hyperedges(source_features)
        
        # 6. Hypergraph Convolution
        source_features_hg = HypergraphConv(source_features, ...)
        
        # 7. Hypergraph Contrastive Loss
        L_hg = InfoNCE(source_features_hg, target_features)
        
        # 8. Total Loss
        L_total = λ_diff * L_diff + λ_hg * L_hg
        
        # 9. Backward
        L_total.backward()
        optimizer.step()
```

## 5. 采样流程（推理）

```python
# 从纯噪声开始
x_T ~ N(0, I)

for t in reversed(range(T)):
    # 预测噪声
    predicted_noise = UNet(x_t, t, condition=source_image)
    
    # 去噪一步
    x_{t-1} = p_sample(x_t, predicted_noise, t)
    
# 返回 x_0 (生成的 target image)
```

## 6. 数据流 Shape 变化

```
Source Image: [B, 3, 256, 256]
    ↓
U-Net Encoder
    ↓
Bottleneck: [B, 512, 16, 16]  (假设)
    ↓
Reshape: [B, 256, 256] → [B, 256, 256]
    ↓
Sample Patches: [B, 64, 256]  (num_patches=64)
    ↓
Soft K-Means: [B, 64, 9] (memberships)
    ↓
HypergraphConv: [B, 64, 256] → [B, 64, 256]
    ↓
Contrastive Loss with Target Features
```

## 7. 后续可扩展点

1. **Object-level Hypergraph**: 使用预训练的 object detector 提取 object-level features，构建更高层次的 hypergraph

2. **Attention Conditioning**: 在 U-Net 的 cross-attention 层中融入 hypergraph 结构信息

3. **Multi-scale Hypergraph**: 在不同分辨率层级构建多个 hypergraph，进行多尺度结构约束

4. **Adaptive Hyperedge Number**: 根据图像内容自适应调整 hyperedge 数量

5. **Domain-specific Hypergraph**: 为 source 和 target domain 分别构建 hypergraph，然后进行跨 domain 对齐

