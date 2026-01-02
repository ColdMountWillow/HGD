"""
Hypergraph Construction 模块
从 patch features 构建 hypergraph 结构

核心思想:
- 将特征图展开为 patch tokens
- 使用 soft k-means 聚类构建 hyperedges
- 每个 hyperedge 对应一个语义簇
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def soft_kmeans_clustering(
    x: torch.Tensor,
    num_clusters: int,
    num_iters: int = 10,
    temperature: float = 0.2,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft K-Means 聚类
    
    与硬聚类不同，soft k-means 允许每个点以不同程度属于多个簇
    
    Args:
        x: (B, N, D) - batch of N points with D dimensions
        num_clusters: K - 簇数量
        num_iters: 迭代次数
        temperature: softmax 温度，越小越接近硬聚类
        eps: 数值稳定性
    
    Returns:
        memberships: (B, N, K) - soft membership 矩阵
        centers: (B, K, D) - 簇中心
    """
    B, N, D = x.shape
    device = x.device
    
    # 随机初始化簇中心 (从数据点中选择)
    indices = torch.randperm(N, device=device)[:num_clusters]
    centers = x[:, indices, :].clone()  # (B, K, D)
    
    for _ in range(num_iters):
        # 计算点到簇中心的距离
        # x: (B, N, D), centers: (B, K, D)
        # dist: (B, N, K)
        dist = torch.cdist(x, centers, p=2)  # Euclidean distance
        
        # Soft assignment (使用负距离的 softmax)
        memberships = F.softmax(-dist / temperature, dim=-1)  # (B, N, K)
        
        # 更新簇中心
        # weights: (B, K, N)
        weights = memberships.permute(0, 2, 1)
        # numerator: (B, K, D) = (B, K, N) @ (B, N, D)
        numerator = torch.bmm(weights, x)
        # denominator: (B, K, 1)
        denominator = weights.sum(dim=-1, keepdim=True).clamp(min=eps)
        centers = numerator / denominator
    
    return memberships, centers


class HypergraphConstructor(nn.Module):
    """
    Hypergraph 构建器
    
    从特征图构建 hypergraph 结构:
    1. 将特征图展开为 patch tokens
    2. 使用 soft k-means 聚类
    3. 构建 hyperedge 表示
    
    Input:
        features: (B, C, H, W) - 特征图 (来自 U-Net 中间层)
    
    Output:
        patch_features: (B, N, D) - N = H*W 个 patch 的特征
        memberships: (B, N, K) - soft membership 矩阵
        hyperedge_features: (B, K, D) - hyperedge 特征 (簇中心)
        incidence_matrix: (B, N, K) - 二值化的关联矩阵 (可选)
    """
    def __init__(
        self,
        num_clusters: int = 9,
        membership_threshold: float = 0.15,
        num_iters: int = 10,
        temperature: float = 0.2,
        project_dim: Optional[int] = None
    ):
        """
        Args:
            num_clusters: hyperedge 数量 (K)
            membership_threshold: 二值化 membership 的阈值
            num_iters: soft k-means 迭代次数
            temperature: soft k-means 温度
            project_dim: 可选的特征投影维度
        """
        super().__init__()
        
        self.num_clusters = num_clusters
        self.membership_threshold = membership_threshold
        self.num_iters = num_iters
        self.temperature = temperature
        
        # 可选的特征投影
        self.projector = None
        if project_dim is not None:
            self.projector = nn.LazyLinear(project_dim)
    
    def forward(
        self, 
        features: torch.Tensor,
        return_binary: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            features: (B, C, H, W) 特征图
            return_binary: 是否返回二值化的关联矩阵
        
        Returns:
            patch_features: (B, N, D) N = H*W
            memberships: (B, N, K) soft membership
            hyperedge_features: (B, K, D) 簇中心
            incidence_matrix: (B, N, K) 二值化关联矩阵 (如果 return_binary=True)
        """
        B, C, H, W = features.shape
        N = H * W
        
        # 展开为 patch tokens: (B, C, H, W) -> (B, N, C)
        patch_features = features.flatten(2).permute(0, 2, 1)  # (B, N, C)
        
        # 可选投影
        if self.projector is not None:
            patch_features = self.projector(patch_features)
        
        D = patch_features.shape[-1]
        
        # Soft K-Means 聚类 (detach 避免梯度流入聚类过程)
        with torch.no_grad():
            memberships, hyperedge_features = soft_kmeans_clustering(
                patch_features.detach(),
                self.num_clusters,
                self.num_iters,
                self.temperature
            )
        
        # 二值化关联矩阵 (可选)
        incidence_matrix = None
        if return_binary:
            incidence_matrix = (memberships > self.membership_threshold).float()
        
        return patch_features, memberships, hyperedge_features, incidence_matrix
    
    def get_hyperedge_to_node_index(
        self, 
        memberships: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取 hyperedge -> node 的索引 (用于消息传递)
        
        Args:
            memberships: (B, N, K)
        
        Returns:
            hyperedge_matrix: (B, K, max_nodes) 每个 hyperedge 包含的节点索引，-1 表示 padding
            node_counts: (B, K) 每个 hyperedge 的实际节点数
        """
        B, N, K = memberships.shape
        device = memberships.device
        
        # 二值化
        binary = memberships > self.membership_threshold
        
        # 初始化
        hyperedge_matrix = torch.full((B, K, N), -1, dtype=torch.long, device=device)
        node_counts = torch.zeros(B, K, dtype=torch.long, device=device)
        
        for b in range(B):
            for k in range(K):
                node_indices = torch.where(binary[b, :, k])[0]
                num_nodes = len(node_indices)
                hyperedge_matrix[b, k, :num_nodes] = node_indices
                node_counts[b, k] = num_nodes
        
        return hyperedge_matrix, node_counts
    
    def get_node_to_hyperedge_index(
        self, 
        memberships: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取 node -> hyperedge 的索引
        
        Args:
            memberships: (B, N, K)
        
        Returns:
            node_hyperedge_matrix: (B, N, max_edges) 每个节点属于的 hyperedge 索引
            edge_counts: (B, N) 每个节点属于的 hyperedge 数量
        """
        B, N, K = memberships.shape
        device = memberships.device
        
        binary = memberships > self.membership_threshold
        max_edges = binary.sum(dim=-1).max().item()
        
        node_hyperedge_matrix = torch.full((B, N, max_edges), -1, dtype=torch.long, device=device)
        edge_counts = torch.zeros(B, N, dtype=torch.long, device=device)
        
        for b in range(B):
            for n in range(N):
                edge_indices = torch.where(binary[b, n, :])[0]
                num_edges = len(edge_indices)
                node_hyperedge_matrix[b, n, :num_edges] = edge_indices
                edge_counts[b, n] = num_edges
        
        return node_hyperedge_matrix, edge_counts


# 测试
if __name__ == "__main__":
    # 测试 soft k-means
    print("Testing soft k-means...")
    x = torch.randn(2, 100, 64)  # 2 batches, 100 points, 64 dims
    memberships, centers = soft_kmeans_clustering(x, num_clusters=9)
    print(f"Input: {x.shape}")
    print(f"Memberships: {memberships.shape}")
    print(f"Centers: {centers.shape}")
    print(f"Membership sum per point: {memberships.sum(dim=-1)[0, :5]}")  # 应该接近 1
    
    # 测试 HypergraphConstructor
    print("\nTesting HypergraphConstructor...")
    constructor = HypergraphConstructor(num_clusters=9)
    features = torch.randn(2, 256, 16, 16)  # 模拟 U-Net 中间特征
    
    patch_feat, memberships, hyperedge_feat, incidence = constructor(features, return_binary=True)
    print(f"Features: {features.shape}")
    print(f"Patch features: {patch_feat.shape}")
    print(f"Memberships: {memberships.shape}")
    print(f"Hyperedge features: {hyperedge_feat.shape}")
    print(f"Incidence matrix: {incidence.shape}")
    
    # 统计每个 hyperedge 的节点数
    nodes_per_edge = (memberships > 0.15).sum(dim=1)  # (B, K)
    print(f"Nodes per hyperedge: {nodes_per_edge[0]}")

