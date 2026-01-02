"""
Conditional Diffusion U-Net
支持图像条件输入，并可以提取中间特征用于 hypergraph 构建
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class SinusoidalPositionEmbeddings(nn.Module):
    """时间步位置编码"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: [B]
        Returns:
            embeddings: [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            time_emb: [B, time_emb_dim]
        """
        h = self.block1(x)
        # 时间嵌入
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    """自注意力块"""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = F.softmax(attn, dim=-1)
        h = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        
        return x + self.proj(h)


class UNet(nn.Module):
    """
    Conditional U-Net for Diffusion
    
    支持：
    1. 时间步条件
    2. 图像条件（source image）
    3. 中间特征提取（用于 hypergraph）
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16, 8),
        time_emb_dim: int = 128,
        condition_channels: int = 3,
        extract_features: bool = True,
        feature_layer: str = "bottleneck"  # "bottleneck" or "mid"
    ):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            base_channels: 基础通道数
            channel_multipliers: 通道倍数
            num_res_blocks: 每个分辨率层的残差块数
            attention_resolutions: 应用注意力的分辨率
            time_emb_dim: 时间嵌入维度
            condition_channels: 条件图像通道数
            extract_features: 是否提取中间特征
            feature_layer: 特征提取层位置
        """
        super().__init__()
        self.extract_features = extract_features
        self.feature_layer = feature_layer
        
        # 时间嵌入
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(condition_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1)
        )
        
        # 输入投影
        self.input_proj = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 下采样
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        channels_list = [base_channels * m for m in channel_multipliers]
        
        for i, (in_ch, out_ch) in enumerate(zip([base_channels] + channels_list[:-1], channels_list)):
            blocks = nn.ModuleList([
                ResidualBlock(in_ch if j == 0 else out_ch, out_ch, time_emb_dim)
                for j in range(num_res_blocks)
            ])
            self.down_blocks.append(blocks)
            
            # 注意力
            if i < len(channel_multipliers) - 1:
                resolution = 2 ** (len(channel_multipliers) - 2 - i)
                if resolution in attention_resolutions:
                    blocks.append(AttentionBlock(out_ch))
            
            # 下采样（除了最后一层）
            if i < len(channel_multipliers) - 1:
                self.down_samples.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            else:
                self.down_samples.append(nn.Identity())
        
        # Bottleneck（用于特征提取）
        mid_ch = channels_list[-1]
        self.mid_block1 = ResidualBlock(mid_ch, mid_ch, time_emb_dim)
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid_block2 = ResidualBlock(mid_ch, mid_ch, time_emb_dim)
        
        # 上采样
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for i, (in_ch, out_ch) in enumerate(zip(channels_list[::-1], [base_channels] + channels_list[-2::-1])):
            blocks = nn.ModuleList([
                ResidualBlock(in_ch + out_ch if j == 0 else out_ch, out_ch, time_emb_dim)
                for j in range(num_res_blocks)
            ])
            self.up_blocks.append(blocks)
            
            # 注意力
            if i > 0:
                resolution = 2 ** (i)
                if resolution in attention_resolutions:
                    blocks.append(AttentionBlock(out_ch))
            
            # 上采样（除了最后一层）
            if i < len(channel_multipliers) - 1:
                self.up_samples.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1)
                ))
            else:
                self.up_samples.append(nn.Identity())
        
        # 输出投影
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 噪声图像 [B, C, H, W]
            timestep: 时间步 [B]
            condition: 条件图像 [B, C, H, W]
        
        Returns:
            noise: 预测的噪声 [B, C, H, W]
        """
        # 时间嵌入
        time_emb = self.time_mlp(timestep)
        
        # 条件编码
        if condition is not None:
            cond_feat = self.condition_encoder(condition)
        else:
            cond_feat = None
        
        # 输入投影
        x = self.input_proj(x)
        if cond_feat is not None:
            x = x + cond_feat
        
        # 下采样
        skip_connections = []
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            for block in blocks:
                if isinstance(block, ResidualBlock):
                    x = block(x, time_emb)
                else:  # AttentionBlock
                    x = block(x)
            skip_connections.append(x)
            x = downsample(x)
        
        # Bottleneck（特征提取点）
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        bottleneck_feat = x  # 保存用于 hypergraph
        x = self.mid_block2(x, time_emb)
        
        # 上采样
        for blocks, upsample, skip in zip(self.up_blocks, self.up_samples, skip_connections[::-1]):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                if isinstance(block, ResidualBlock):
                    x = block(x, time_emb)
                else:  # AttentionBlock
                    x = block(x)
        
        # 输出
        x = self.output_norm(x)
        x = F.siLU(x)
        x = self.output_conv(x)
        
        return x
    
    def extract_patch_features(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        num_patches: int = 64
    ) -> torch.Tensor:
        """
        提取 patch features 用于 hypergraph 构建
        
        Args:
            x: 噪声图像 [B, C, H, W]
            timestep: 时间步 [B]
            condition: 条件图像 [B, C, H, W]
            num_patches: 采样 patch 数量
        
        Returns:
            patch_features: [B, num_patches, feature_dim]
        """
        # 时间嵌入
        time_emb = self.time_mlp(timestep)
        
        # 条件编码
        if condition is not None:
            cond_feat = self.condition_encoder(condition)
        else:
            cond_feat = None
        
        # 输入投影
        x = self.input_proj(x)
        if cond_feat is not None:
            x = x + cond_feat
        
        # 下采样到 bottleneck
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            for block in blocks:
                if isinstance(block, ResidualBlock):
                    x = block(x, time_emb)
                else:
                    x = block(x)
            x = downsample(x)
        
        # 提取 bottleneck 特征
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        
        # 将特征图 reshape 为 patches
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = x.reshape(B, H * W, C)  # [B, H*W, C]
        
        # 随机采样 patches
        if num_patches > 0 and num_patches < H * W:
            indices = torch.randperm(H * W, device=x.device)[:num_patches]
            x = x[:, indices, :]
        
        return x  # [B, num_patches, C]

