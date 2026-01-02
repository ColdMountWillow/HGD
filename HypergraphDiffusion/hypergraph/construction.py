"""
Hypergraph 构建模块
基于 soft k-means 聚类构建 hyperedges
"""
import torch
import torch.nn.functional as F
from typing import Tuple


def soft_k_means(
    x: torch.Tensor,
    n_clusters: int,
    epsilon: float = 5e-2,
    max_iter: int = 100,
    temperature: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft k-means 聚类
    
    Args:
        x: 特征向量 [B, num_points, num_dims]
        n_clusters: 聚类数量（hyperedge 数量）
        epsilon: 收敛阈值
        max_iter: 最大迭代次数
        temperature: 温度参数，控制 soft assignment 的 softness
    
    Returns:
        memberships: 隶属度矩阵 [B, num_points, n_clusters]
        centers: 聚类中心 [B, num_dims, n_clusters]
    """
    device = x.device
    batch_size, num_points, num_dims = x.shape
    
    # 随机初始化聚类中心
    indices = torch.randperm(num_points, device=device)[:n_clusters]
    centers = x[:, indices, :].transpose(1, 2)  # [B, num_dims, n_clusters]
    
    for iteration in range(max_iter):
        # 计算距离
        dist = torch.cdist(x, centers.transpose(1, 2), p=2)  # [B, num_points, n_clusters]
        
        # Soft assignment
        memberships = F.softmax(-dist / temperature, dim=-1)  # [B, num_points, n_clusters]
        
        # 更新聚类中心
        weights = memberships.transpose(1, 2)  # [B, n_clusters, num_points]
        numerator = torch.bmm(weights, x)  # [B, n_clusters, num_dims]
        denominator = weights.sum(dim=-1, keepdim=True)  # [B, n_clusters, 1]
        new_centers = numerator / (denominator + 1e-8)
        
        # 检查收敛
        if torch.norm(new_centers - centers) < epsilon:
            break
        
        centers = new_centers.transpose(1, 2)  # [B, num_dims, n_clusters]
    
    # 最终计算 memberships
    dist = torch.cdist(x, centers.transpose(1, 2), p=2)
    memberships = F.softmax(-dist / temperature, dim=-1)
    
    return memberships, centers


def construct_hyperedges(
    patch_features: torch.Tensor,
    num_clusters: int,
    threshold: float = 0.15,
    temperature: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从 patch features 构建 hyperedges
    
    Args:
        patch_features: Patch 特征 [B, num_patches, feature_dim]
        num_clusters: Hyperedge 数量
        threshold: 隶属度阈值，用于确定 point 是否属于 hyperedge
        temperature: Soft k-means 温度参数
    
    Returns:
        hyperedge_matrix: [B, n_clusters, num_patches] - 每个 hyperedge 包含的 points（用 -1 padding）
        point_hyperedge_index: [B, num_patches, max_edges_per_point] - 每个 point 属于的 hyperedges（用 -1 padding）
        hyperedge_features: [B, feature_dim, n_clusters] - 每个 hyperedge 的特征（聚类中心）
    """
    device = patch_features.device
    batch_size, num_patches, feature_dim = patch_features.shape
    
    # 使用 soft k-means 聚类
    memberships, centers = soft_k_means(
        patch_features,
        n_clusters=num_clusters,
        temperature=temperature
    )
    
    # 构建 hyperedge_matrix: [B, n_clusters, num_patches]
    # 每个 hyperedge 包含的 points 索引
    hyperedge_matrix = -torch.ones(batch_size, num_clusters, num_patches, dtype=torch.long, device=device)
    
    for c in range(num_clusters):
        mask = memberships[:, :, c] > threshold  # [B, num_patches]
        for b in range(batch_size):
            idxs = torch.where(mask[b])[0]
            if len(idxs) > 0:
                hyperedge_matrix[b, c, :len(idxs)] = idxs
    
    # 构建 point_hyperedge_index: [B, num_patches, max_edges_per_point]
    # 每个 point 属于的 hyperedges
    max_edges_per_point = (memberships > threshold).sum(dim=-1).max().item()
    if max_edges_per_point == 0:
        max_edges_per_point = 1
    
    point_hyperedge_index = -torch.ones(
        batch_size, num_patches, max_edges_per_point,
        dtype=torch.long, device=device
    )
    
    for b in range(batch_size):
        for p in range(num_patches):
            idxs = torch.where(memberships[b, p, :] > threshold)[0]
            if len(idxs) > 0:
                point_hyperedge_index[b, p, :len(idxs)] = idxs
    
    # hyperedge_features 就是聚类中心
    hyperedge_features = centers  # [B, feature_dim, n_clusters]
    
    return hyperedge_matrix, point_hyperedge_index, hyperedge_features

