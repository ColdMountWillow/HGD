"""
Hypergraph Neural Network (HGNN)
用于 hypergraph 上的信息传播
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def batched_index_select(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    批量索引选择
    
    Args:
        x: [B, C, N, 1] 或 [B, C, N]
        indices: [B, K, M] -1 表示 padding
    
    Returns:
        selected: [B, C, K, M]
    """
    B, C = x.shape[0], x.shape[1]
    
    if x.dim() == 4:
        x = x.squeeze(-1)  # [B, C, N]
    
    N = x.shape[2]
    K, M = indices.shape[1], indices.shape[2]
    
    # 创建 mask
    mask = indices >= 0  # [B, K, M]
    
    # 将 -1 替换为 0（用于索引，但会被 mask 掉）
    indices_safe = indices.clamp(min=0)
    
    # 批量索引
    batch_indices = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, K, M)
    selected = x[batch_indices, :, indices_safe]  # [B, C, K, M]
    
    # 应用 mask
    selected = selected * mask.unsqueeze(1).float()
    
    return selected.unsqueeze(-1)  # [B, C, K, M, 1]


class HypergraphConv(nn.Module):
    """
    Hypergraph Convolution Layer
    
    实现 node-to-hyperedge 和 hyperedge-to-node 的信息传播
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True
    ):
        """
        Args:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            bias: 是否使用 bias
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Node to hyperedge transformation
        self.node_to_hyperedge = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )
        
        # Hyperedge to node transformation
        self.hyperedge_to_node = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Learnable epsilon (类似 GIN)
        self.eps = nn.Parameter(torch.tensor(0.0))
    
    def forward(
        self,
        node_features: torch.Tensor,
        hyperedge_matrix: torch.Tensor,
        point_hyperedge_index: torch.Tensor,
        hyperedge_centers: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_features: Node 特征 [B, C, N, 1] 或 [B, C, N]
            hyperedge_matrix: [B, K, N] - 每个 hyperedge 包含的 nodes
            point_hyperedge_index: [B, N, M] - 每个 node 属于的 hyperedges
            hyperedge_centers: Hyperedge 中心特征 [B, C, K]
        
        Returns:
            updated_node_features: [B, out_channels, N, 1]
        """
        device = node_features.device
        B = node_features.shape[0]
        
        # 确保 node_features 是 [B, C, N, 1] 格式
        if node_features.dim() == 3:
            node_features = node_features.unsqueeze(-1)  # [B, C, N, 1]
        
        C, N = node_features.shape[1], node_features.shape[2]
        K = hyperedge_matrix.shape[1]
        
        # Step 1: Node to Hyperedge aggregation
        # 从 hyperedge_matrix 中提取每个 hyperedge 的 nodes
        node_features_for_hyperedges = batched_index_select(node_features, hyperedge_matrix)
        # node_features_for_hyperedges: [B, C, K, N, 1]
        
        # 聚合：对每个 hyperedge 的 nodes 求和
        aggregated_hyperedge_features = node_features_for_hyperedges.sum(dim=3, keepdim=True)
        # [B, C, K, 1, 1]
        
        # Reshape 为 [B, K, C] 用于 MLP
        aggregated_hyperedge_features = aggregated_hyperedge_features.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        # [B, K, C]
        
        # Node to hyperedge transformation
        aggregated_hyperedge_features = self.node_to_hyperedge(aggregated_hyperedge_features)
        # [B, K, C]
        
        # 添加 hyperedge center features
        hyperedge_centers_T = hyperedge_centers.permute(0, 2, 1)  # [B, K, C]
        aggregated_hyperedge_features = aggregated_hyperedge_features + (1 + self.eps) * hyperedge_centers_T
        # [B, K, C]
        
        # Step 2: Hyperedge to Node aggregation
        # 将 hyperedge features reshape 为 [B, C, K, 1] 用于 batched_index_select
        hyperedge_features_for_select = aggregated_hyperedge_features.permute(0, 2, 1).unsqueeze(-1)
        # [B, C, K, 1]
        
        # 从 point_hyperedge_index 中提取每个 node 的 hyperedges
        hyperedge_features_for_nodes = batched_index_select(hyperedge_features_for_select, point_hyperedge_index)
        # [B, C, N, M, 1]
        
        # 聚合：对每个 node 的 hyperedges 求和
        aggregated_node_features = hyperedge_features_for_nodes.sum(dim=3, keepdim=True)
        # [B, C, N, 1, 1]
        
        # Reshape 为 [B, N, C] 用于 MLP
        aggregated_node_features = aggregated_node_features.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        # [B, N, C]
        
        # Hyperedge to node transformation
        updated_node_features = self.hyperedge_to_node(aggregated_node_features)
        # [B, N, out_channels]
        
        # Reshape 回 [B, out_channels, N, 1]
        updated_node_features = updated_node_features.permute(0, 2, 1).unsqueeze(-1)
        # [B, out_channels, N, 1]
        
        return updated_node_features

