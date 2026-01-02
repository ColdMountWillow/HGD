"""
Condition Encoder
用于编码 source image 作为条件

注：在简化版中，我们直接使用 concat 方式
这个模块提供可选的预编码功能（如使用预训练 encoder）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleEncoder(nn.Module):
    """
    简单的卷积编码器
    将图像编码为特征表示
    
    Input: (B, C, H, W)
    Output: (B, out_channels, H, W)
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        num_layers: int = 3
    ):
        super().__init__()
        
        layers = []
        ch = in_channels
        
        for i in range(num_layers):
            out_ch = out_channels if i == num_layers - 1 else min(out_channels, 32 * (2 ** i))
            layers.extend([
                nn.Conv2d(ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU()
            ])
            ch = out_ch
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ConditionEncoder(nn.Module):
    """
    条件编码器
    
    支持多种编码方式:
    - "concat": 直接返回原图 (用于 U-Net concat)
    - "simple": 使用简单卷积编码器
    - "pretrained": 使用预训练模型 (未实现)
    
    Input: (B, C, H, W)
    Output: (B, out_channels, H, W)
    """
    def __init__(
        self,
        mode: str = "concat",
        in_channels: int = 3,
        out_channels: int = 3,
        **kwargs
    ):
        super().__init__()
        
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if mode == "concat":
            # 直接返回原图
            self.encoder = nn.Identity()
        elif mode == "simple":
            self.encoder = SimpleEncoder(in_channels, out_channels, **kwargs)
        else:
            raise ValueError(f"Unknown condition encoder mode: {mode}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) source image
        
        Returns:
            (B, out_channels, H, W) encoded condition
        """
        return self.encoder(x)


# 测试
if __name__ == "__main__":
    # Test concat mode
    encoder = ConditionEncoder(mode="concat")
    x = torch.randn(2, 3, 128, 128)
    out = encoder(x)
    print(f"Concat mode - Input: {x.shape}, Output: {out.shape}")
    
    # Test simple mode
    encoder = ConditionEncoder(mode="simple", out_channels=64, num_layers=3)
    out = encoder(x)
    print(f"Simple mode - Input: {x.shape}, Output: {out.shape}")

