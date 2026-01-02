"""
Hypergraph Neural Network (HGNN) 模块
实现 hypergraph 上的消息传递

参考: STNHCL 的 hypergraph convolution 实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class HypergraphConv(nn.Module):
    """
    Hypergraph Convolution Layer
    
    基于 GIN (Graph Isomorphism Network) 风格的消息传递:
    1. Node -> Hyperedge: 聚合节点特征到 hyperedge
    2. Hyperedge -> Node: 聚合 hyperedge 特征回节点
    
    Input:
        node_features: (B, N, D) - 节点特征
        memberships: (B, N, K) - soft membership 矩阵
        hyperedge_centers: (B, K, D) - hyperedge 中心特征
    
    Output:
        updated_node_features: (B, N, D_out)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Node -> Hyperedge transformation
        self.node_to_edge = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=bias),
            nn.ReLU()
        )
        
        # Hyperedge -> Node transformation
        self.edge_to_node = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            nn.ReLU()
        )
        
        # Learnable epsilon for GIN-style aggregation
        self.eps = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        node_features: torch.Tensor,
        memberships: torch.Tensor,
        hyperedge_centers: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            node_features: (B, N, D_in)
            memberships: (B, N, K) - soft membership
            hyperedge_centers: (B, K, D_in) - 可选，如果提供则加入 hyperedge 特征
        
        Returns:
            updated_features: (B, N, D_out)
        """
        B, N, D = node_features.shape
        K = memberships.shape[-1]
        
        # Step 1: Node -> Hyperedge aggregation
        # 使用 membership 作为权重聚合节点特征
        # (B, K, N) @ (B, N, D) -> (B, K, D)
        weights_n2e = memberships.permute(0, 2, 1)  # (B, K, N)
        hyperedge_features = torch.bmm(weights_n2e, node_features)  # (B, K, D)
        
        # 归一化
        weights_sum = weights_n2e.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, K, 1)
        hyperedge_features = hyperedge_features / weights_sum
        
        # Transform
        hyperedge_features = self.node_to_edge(hyperedge_features)
        
        # 可选: 加入原始 hyperedge center 特征
        if hyperedge_centers is not None:
            hyperedge_features = hyperedge_features + (1 + self.eps) * hyperedge_centers
        
        # Step 2: Hyperedge -> Node aggregation
        # 使用 membership 作为权重聚合 hyperedge 特征回节点
        # (B, N, K) @ (B, K, D) -> (B, N, D)
        aggregated_features = torch.bmm(memberships, hyperedge_features)  # (B, N, D)
        
        # 归一化
        weights_sum_e2n = memberships.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, N, 1)
        aggregated_features = aggregated_features / weights_sum_e2n
        
        # Transform
        output = self.edge_to_node(aggregated_features)
        
        return output


class HypergraphEncoder(nn.Module):
    """
    Hypergraph Encoder
    
    多层 hypergraph convolution + 最终投影
    用于编码 patch-level 结构信息
    
    Input:
        node_features: (B, N, D) - 节点特征
        memberships: (B, N, K) - soft membership
        hyperedge_centers: (B, K, D) - hyperedge 中心
    
    Output:
        encoded_features: (B, N, D_out) - 编码后的特征
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        normalize: bool = True
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.normalize = normalize
        
        # Hypergraph convolution layers
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = hidden_channels if i < num_layers - 1 else out_channels
            self.convs.append(HypergraphConv(in_ch, out_ch))
        
        # L2 normalization
        if normalize:
            self.l2norm = lambda x: F.normalize(x, p=2, dim=-1)
    
    def forward(
        self,
        node_features: torch.Tensor,
        memberships: torch.Tensor,
        hyperedge_centers: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            node_features: (B, N, D_in)
            memberships: (B, N, K)
            hyperedge_centers: (B, K, D_in)
        
        Returns:
            (B, N, D_out)
        """
        h = node_features
        
        for i, conv in enumerate(self.convs):
            # 只在第一层使用 hyperedge centers
            centers = hyperedge_centers if i == 0 else None
            h = conv(h, memberships, centers)
        
        if self.normalize:
            h = self.l2norm(h)
        
        return h


class DualHypergraphEncoder(nn.Module):
    """
    双分支 Hypergraph Encoder
    
    用于同时处理 source 和 target domain 的特征
    共享 hypergraph 结构但分别编码
    
    这对于 contrastive loss 很重要：
    - 使用相同的 hypergraph 结构
    - 但分别处理两个 domain 的特征
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2
    ):
        super().__init__()
        
        # 两个独立的 encoder (不共享权重)
        self.encoder_source = HypergraphEncoder(
            in_channels, hidden_channels, out_channels, num_layers
        )
        self.encoder_target = HypergraphEncoder(
            in_channels, hidden_channels, out_channels, num_layers
        )
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        memberships: torch.Tensor,
        hyperedge_centers: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            source_features: (B, N, D) - source domain patch features
            target_features: (B, N, D) - target domain patch features
            memberships: (B, N, K) - 共享的 hypergraph 结构 (从 target 构建)
            hyperedge_centers: (B, K, D)
        
        Returns:
            source_encoded: (B, N, D_out)
            target_encoded: (B, N, D_out)
        """
        source_encoded = self.encoder_source(source_features, memberships, hyperedge_centers)
        target_encoded = self.encoder_target(target_features, memberships, hyperedge_centers)
        
        return source_encoded, target_encoded


# 测试
if __name__ == "__main__":
    print("Testing HypergraphConv...")
    conv = HypergraphConv(64, 128)
    
    B, N, D, K = 2, 256, 64, 9
    node_feat = torch.randn(B, N, D)
    memberships = F.softmax(torch.randn(B, N, K), dim=-1)
    centers = torch.randn(B, K, D)
    
    out = conv(node_feat, memberships, centers)
    print(f"Input: {node_feat.shape}")
    print(f"Output: {out.shape}")
    
    print("\nTesting HypergraphEncoder...")
    encoder = HypergraphEncoder(64, 128, 256, num_layers=2)
    out = encoder(node_feat, memberships, centers)
    print(f"Encoded output: {out.shape}")
    print(f"Output norm: {out.norm(dim=-1).mean():.4f}")  # 应该接近 1 (因为 L2 normalize)
    
    print("\nTesting DualHypergraphEncoder...")
    dual_encoder = DualHypergraphEncoder(64, 128, 256)
    source_feat = torch.randn(B, N, D)
    target_feat = torch.randn(B, N, D)
    
    src_enc, tgt_enc = dual_encoder(source_feat, target_feat, memberships, centers)
    print(f"Source encoded: {src_enc.shape}")
    print(f"Target encoded: {tgt_enc.shape}")

