"""
Diffusion 损失函数
标准 DDPM noise prediction loss
"""
import torch
import torch.nn as nn


class DiffusionLoss(nn.Module):
    """
    Diffusion 噪声预测损失
    """
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predicted_noise: 模型预测的噪声 [B, C, H, W]
            target_noise: 真实噪声 [B, C, H, W]
        
        Returns:
            loss: 标量损失值
        """
        return self.mse_loss(predicted_noise, target_noise)

