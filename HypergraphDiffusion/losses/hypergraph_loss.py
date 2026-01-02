"""
Hypergraph 对比损失
约束 source 和 target domain 的 patch-level 结构一致性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HypergraphContrastiveLoss(nn.Module):
    """
    Hypergraph 对比损失
    
    使用 InfoNCE 损失约束 source 和 target domain 的 patch features
    在 hypergraph 结构约束下的一致性
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: 对比学习温度参数
        """
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 hypergraph 对比损失
        
        Args:
            source_features: Source domain 的 patch features [B, num_patches, feature_dim]
            target_features: Target domain 的 patch features [B, num_patches, feature_dim]
        
        Returns:
            loss: 标量损失值
        """
        B, N, C = source_features.shape
        
        # L2 归一化
        source_features = F.normalize(source_features, p=2, dim=-1)  # [B, N, C]
        target_features = F.normalize(target_features, p=2, dim=-1)  # [B, N, C]
        
        # 计算 positive pairs (对应位置的 patches)
        pos_sim = torch.einsum('bnc,bnc->bn', [source_features, target_features])
        pos_sim = pos_sim.unsqueeze(-1)  # [B, N, 1]
        
        # 计算 negative pairs
        # 对于每个 source patch，所有 target patches 都是 negatives（除了对应位置的）
        # 简化实现：使用 batch 内的其他 patches 作为 negatives
        
        # Reshape 为 [B*N, C]
        source_flat = source_features.view(B * N, C)  # [B*N, C]
        target_flat = target_features.view(B * N, C)  # [B*N, C]
        
        # 计算所有 pairs 的相似度
        all_sim = torch.mm(source_flat, target_flat.t())  # [B*N, B*N]
        
        # 创建 mask：对角线为 positive pairs
        mask = torch.eye(B * N, device=source_features.device, dtype=torch.bool)
        all_sim.masked_fill_(mask, -1e9)  # 将 positive pairs 设为很小的值
        
        # 构建 logits: [positive, negative_1, negative_2, ...]
        pos_sim_flat = pos_sim.view(B * N, 1)  # [B*N, 1]
        logits = torch.cat([pos_sim_flat, all_sim], dim=1) / self.temperature
        # [B*N, B*N+1]
        
        # 标签：positive pair 在位置 0
        labels = torch.zeros(B * N, dtype=torch.long, device=source_features.device)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        return loss

