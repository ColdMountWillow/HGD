"""
Diffusion Loss
标准的噪声预测损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiffusionLoss(nn.Module):
    """
    Diffusion 训练损失
    
    支持:
    - MSE loss (默认)
    - L1 loss
    - Weighted loss (可选的 timestep 加权)
    """
    def __init__(
        self,
        loss_type: str = "mse",
        reduction: str = "mean"
    ):
        """
        Args:
            loss_type: "mse" 或 "l1"
            reduction: "mean", "sum", "none"
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            noise_pred: (B, C, H, W) 预测的噪声
            noise_target: (B, C, H, W) 真实噪声
            weights: (B,) 可选的样本权重
        
        Returns:
            loss: scalar
        """
        if weights is None:
            return self.loss_fn(noise_pred, noise_target)
        else:
            # Per-sample loss
            loss_per_sample = F.mse_loss(noise_pred, noise_target, reduction='none')
            loss_per_sample = loss_per_sample.mean(dim=[1, 2, 3])  # (B,)
            
            # Weighted mean
            weights = weights / weights.sum()
            return (loss_per_sample * weights).sum()


# 测试
if __name__ == "__main__":
    loss_fn = DiffusionLoss(loss_type="mse")
    
    pred = torch.randn(4, 3, 64, 64)
    target = torch.randn(4, 3, 64, 64)
    
    loss = loss_fn(pred, target)
    print(f"Diffusion loss: {loss.item():.4f}")
    
    # With weights
    weights = torch.tensor([1.0, 2.0, 1.0, 1.0])
    loss_weighted = loss_fn(pred, target, weights)
    print(f"Weighted loss: {loss_weighted.item():.4f}")

