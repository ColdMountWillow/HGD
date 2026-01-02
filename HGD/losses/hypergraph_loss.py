"""
Hypergraph Contrastive Loss
核心创新：约束跨 domain 的 patch-level 结构一致性

设计思路:
- 构建 hypergraph 从 diffusion 中间特征
- 使用 HGNN 编码 patch 结构信息
- 通过 contrastive loss 约束 source 和 target 的结构保持一致
- 允许风格/纹理变化，但组织拓扑不应崩坏
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypergraph.construction import HypergraphConstructor
from hypergraph.hgnn import DualHypergraphEncoder


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (也称为 NT-Xent)
    
    用于 node-wise contrastive learning:
    - positive pair: 同一位置的 source 和 target patch
    - negative pairs: 不同位置的 patches
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features_a: (B*N, D) - source domain 特征
            features_b: (B*N, D) - target domain 特征
        
        Returns:
            loss: scalar
        """
        # L2 normalize
        features_a = F.normalize(features_a, p=2, dim=-1)
        features_b = F.normalize(features_b, p=2, dim=-1)
        
        N = features_a.shape[0]
        
        # Positive logits: 对应位置的相似度
        pos_logits = (features_a * features_b).sum(dim=-1, keepdim=True)  # (N, 1)
        
        # Negative logits: 所有 cross-domain 相似度
        neg_logits = torch.mm(features_a, features_b.t())  # (N, N)
        
        # 构建 logits: [pos, neg_1, neg_2, ..., neg_N]
        # 但这里我们用更简洁的方式: 直接把对角线当作 positive
        logits = neg_logits / self.temperature  # (N, N)
        
        # Labels: 每个样本的 positive 在对角线位置
        labels = torch.arange(N, device=features_a.device)
        
        # Cross entropy loss
        loss = self.criterion(logits, labels)
        
        return loss


class WeightedInfoNCELoss(nn.Module):
    """
    带权重的 InfoNCE Loss
    
    参考 STNHCL 的双正态分布加权策略:
    - 对于相似度高的 negative pairs 给予较高权重 (hard negatives)
    - 使用正态分布 PDF 作为权重函数
    """
    def __init__(
        self,
        temperature: float = 0.07,
        weight_mu: float = 0.7,
        weight_sigma: float = 1.0
    ):
        super().__init__()
        self.temperature = temperature
        self.weight_mu = weight_mu
        self.weight_sigma = weight_sigma
        self.criterion = nn.CrossEntropyLoss()
    
    def gaussian_weight(self, similarities: torch.Tensor) -> torch.Tensor:
        """
        使用高斯 PDF 计算权重
        
        Args:
            similarities: (N, N) 相似度矩阵
        
        Returns:
            weights: (N, N) 权重矩阵
        """
        # Gaussian PDF
        weights = (1.0 / (self.weight_sigma * math.sqrt(2 * math.pi))) * \
                  torch.exp(-(similarities - self.weight_mu) ** 2 / (2 * self.weight_sigma ** 2))
        
        # Normalize per row
        weights = weights / (weights.mean(dim=-1, keepdim=True) + 1e-6)
        
        return weights
    
    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features_a: (N, D) - source domain 特征
            features_b: (N, D) - target domain 特征
        
        Returns:
            loss: scalar
        """
        # L2 normalize
        features_a = F.normalize(features_a, p=2, dim=-1)
        features_b = F.normalize(features_b, p=2, dim=-1)
        
        N = features_a.shape[0]
        device = features_a.device
        
        # Positive logits
        pos_logits = (features_a * features_b).sum(dim=-1, keepdim=True)  # (N, 1)
        
        # Negative logits
        neg_logits = torch.mm(features_a, features_b.t())  # (N, N)
        
        # 计算权重 (detach 避免梯度)
        with torch.no_grad():
            weights = self.gaussian_weight(neg_logits)
        
        # 加权 negative logits
        weighted_neg_logits = neg_logits * weights
        
        # Mask diagonal (positive pairs)
        diagonal_mask = torch.eye(N, device=device, dtype=torch.bool)
        weighted_neg_logits.masked_fill_(diagonal_mask, -10.0)
        
        # Concat pos and neg
        logits = torch.cat([pos_logits, weighted_neg_logits], dim=1)  # (N, 1+N)
        logits = logits / self.temperature
        
        # Labels: positive 在第 0 位
        labels = torch.zeros(N, device=device, dtype=torch.long)
        
        loss = self.criterion(logits, labels)
        
        return loss


class HypergraphContrastiveLoss(nn.Module):
    """
    Hypergraph Contrastive Loss (核心模块)
    
    完整的 hypergraph-based structure consistency loss:
    1. 从中间特征构建 hypergraph
    2. 使用 HGNN 编码结构信息
    3. Node-wise contrastive loss 约束结构一致性
    
    Input:
        source_features: (B, C, H, W) - source domain 的中间特征
        target_features: (B, C, H, W) - target domain 的中间特征
    
    Output:
        loss: scalar - contrastive loss
    """
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_clusters: int = 9,
        num_hgnn_layers: int = 2,
        temperature: float = 0.07,
        use_weighted_loss: bool = True,
        membership_threshold: float = 0.15
    ):
        """
        Args:
            feature_dim: 输入特征维度
            hidden_dim: HGNN 隐藏维度
            output_dim: 输出特征维度
            num_clusters: hyperedge 数量
            num_hgnn_layers: HGNN 层数
            temperature: contrastive loss 温度
            use_weighted_loss: 是否使用加权 InfoNCE
            membership_threshold: soft clustering 阈值
        """
        super().__init__()
        
        # Hypergraph 构建器
        self.hypergraph_constructor = HypergraphConstructor(
            num_clusters=num_clusters,
            membership_threshold=membership_threshold
        )
        
        # 双分支 HGNN encoder
        self.hgnn_encoder = DualHypergraphEncoder(
            in_channels=feature_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=num_hgnn_layers
        )
        
        # Contrastive loss
        if use_weighted_loss:
            self.contrastive_loss = WeightedInfoNCELoss(temperature)
        else:
            self.contrastive_loss = InfoNCELoss(temperature)
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            source_features: (B, C, H, W) - source domain 中间特征
            target_features: (B, C, H, W) - target domain 中间特征
        
        Returns:
            loss: scalar
            info: dict 包含调试信息
        """
        B, C, H, W = source_features.shape
        N = H * W  # patch 数量
        
        # 构建 hypergraph (从 target features，因为 target 是我们想生成的)
        patch_features_t, memberships, hyperedge_centers, _ = self.hypergraph_constructor(
            target_features
        )
        
        # 获取 source patch features
        patch_features_s = source_features.flatten(2).permute(0, 2, 1)  # (B, N, C)
        
        # HGNN 编码
        source_encoded, target_encoded = self.hgnn_encoder(
            patch_features_s, 
            patch_features_t,
            memberships,
            hyperedge_centers
        )
        
        # Flatten for contrastive loss
        source_flat = source_encoded.reshape(B * N, -1)  # (B*N, D)
        target_flat = target_encoded.reshape(B * N, -1)  # (B*N, D)
        
        # Bidirectional contrastive loss
        loss_s2t = self.contrastive_loss(source_flat, target_flat)
        loss_t2s = self.contrastive_loss(target_flat, source_flat)
        loss = (loss_s2t + loss_t2s) / 2
        
        # 调试信息
        info = {
            'loss_s2t': loss_s2t.item(),
            'loss_t2s': loss_t2s.item(),
            'num_patches': N,
            'num_hyperedges': memberships.shape[-1],
            'avg_membership': memberships.mean().item()
        }
        
        return loss, info


class HypergraphStructureLoss(nn.Module):
    """
    简化版 Hypergraph Structure Loss
    
    不使用 HGNN，直接在 hypergraph 上计算结构一致性:
    - 比较 source 和 target 在相同 hypergraph 下的 membership 分布
    - 更轻量，但可能不如 contrastive loss 有效
    """
    def __init__(
        self,
        num_clusters: int = 9,
        membership_threshold: float = 0.15
    ):
        super().__init__()
        
        self.hypergraph_constructor = HypergraphConstructor(
            num_clusters=num_clusters,
            membership_threshold=membership_threshold
        )
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算 membership 分布的 KL divergence
        """
        # 分别构建 hypergraph
        _, membership_s, _, _ = self.hypergraph_constructor(source_features)
        _, membership_t, _, _ = self.hypergraph_constructor(target_features)
        
        # KL divergence between membership distributions
        # 加 eps 避免 log(0)
        eps = 1e-6
        membership_s = membership_s.clamp(min=eps)
        membership_t = membership_t.clamp(min=eps)
        
        # Forward KL: KL(P || Q) = sum(P * log(P/Q))
        kl_forward = (membership_t * (membership_t.log() - membership_s.log())).sum(dim=-1).mean()
        kl_backward = (membership_s * (membership_s.log() - membership_t.log())).sum(dim=-1).mean()
        
        loss = (kl_forward + kl_backward) / 2
        
        info = {
            'kl_forward': kl_forward.item(),
            'kl_backward': kl_backward.item()
        }
        
        return loss, info


# 测试
if __name__ == "__main__":
    print("Testing InfoNCELoss...")
    loss_fn = InfoNCELoss(temperature=0.07)
    
    feat_a = torch.randn(256, 128)
    feat_b = torch.randn(256, 128)
    
    loss = loss_fn(feat_a, feat_b)
    print(f"InfoNCE loss: {loss.item():.4f}")
    
    print("\nTesting WeightedInfoNCELoss...")
    weighted_loss_fn = WeightedInfoNCELoss(temperature=0.07)
    loss = weighted_loss_fn(feat_a, feat_b)
    print(f"Weighted InfoNCE loss: {loss.item():.4f}")
    
    print("\nTesting HypergraphContrastiveLoss...")
    hg_loss = HypergraphContrastiveLoss(
        feature_dim=256,
        hidden_dim=128,
        output_dim=128,
        num_clusters=9
    )
    
    source_feat = torch.randn(2, 256, 16, 16)
    target_feat = torch.randn(2, 256, 16, 16)
    
    loss, info = hg_loss(source_feat, target_feat)
    print(f"Hypergraph contrastive loss: {loss.item():.4f}")
    print(f"Info: {info}")
    
    print("\nTesting HypergraphStructureLoss...")
    struct_loss = HypergraphStructureLoss(num_clusters=9)
    loss, info = struct_loss(source_feat, target_feat)
    print(f"Structure loss: {loss.item():.4f}")
    print(f"Info: {info}")

