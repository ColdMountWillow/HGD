"""
Conditional U-Net for Diffusion
简化版实现，支持 image conditioning 和 domain embedding
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embedding
    
    Args:
        timesteps: (B,) tensor of timesteps
        embedding_dim: 嵌入维度
    
    Returns:
        (B, embedding_dim) tensor
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    """GroupNorm with float32 precision"""
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class ResBlock(nn.Module):
    """
    Residual Block with timestep embedding injection
    
    Input: (B, C_in, H, W)
    Output: (B, C_out, H, W)
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        time_emb_dim: int,
        dropout: float = 0.0,
        up: bool = False,
        down: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv
        self.norm1 = GroupNorm32(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_emb_proj = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Second conv
        self.norm2 = GroupNorm32(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()
            
        # Up/Down sampling
        self.up = up
        self.down = down
        if up:
            self.h_upd = nn.Upsample(scale_factor=2, mode='nearest')
            self.x_upd = nn.Upsample(scale_factor=2, mode='nearest')
        elif down:
            self.h_upd = nn.AvgPool2d(2)
            self.x_upd = nn.AvgPool2d(2)
        else:
            self.h_upd = self.x_upd = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)
            time_emb: (B, time_emb_dim)
        Returns:
            (B, C_out, H', W') - H', W' depends on up/down
        """
        h = self.norm1(x)
        h = F.silu(h)
        
        if self.up or self.down:
            h = self.h_upd(h)
            x = self.x_upd(x)
            
        h = self.conv1(h)
        
        # Add time embedding
        h = h + self.time_emb_proj(time_emb)[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip_conv(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block
    
    Input/Output: (B, C, H, W)
    """
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = GroupNorm32(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (B, heads, head_dim, HW)
        
        # Attention
        q = q.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = F.softmax(attn, dim=-1)
        
        h = torch.matmul(attn, v)  # (B, heads, HW, head_dim)
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        return x + self.proj(h)


class ConditionalUNet(nn.Module):
    """
    Conditional U-Net for Diffusion
    
    支持:
    - Image conditioning (concat 方式)
    - Domain embedding
    - 中间特征提取 (用于 hypergraph)
    
    Input:
        x: (B, C, H, W) - noisy image
        t: (B,) - timesteps
        cond: (B, C, H, W) - condition image (source stain)
        domain_id: (B,) - optional domain embedding
    
    Output:
        noise_pred: (B, C, H, W) - predicted noise
        mid_features: (B, C_mid, H_mid, W_mid) - 中间特征 (可选)
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.0,
        num_heads: int = 8,
        num_domains: int = 2,
        domain_embed_dim: int = 128,
        image_size: int = 256,
        return_mid_features: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.return_mid_features = return_mid_features
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            Swish(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Domain embedding (optional)
        self.domain_embed = nn.Embedding(num_domains, domain_embed_dim)
        self.domain_proj = nn.Linear(domain_embed_dim, time_emb_dim)
        
        # Input: concat x + condition
        # 因此输入通道数是 in_channels * 2
        concat_channels = in_channels * 2
        
        # Initial conv
        self.input_conv = nn.Conv2d(concat_channels, model_channels, 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        
        channels = [model_channels]
        ch = model_channels
        ds = 1  # current downsampling factor
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResBlock(ch, out_ch, time_emb_dim, dropout)
                )
                ch = out_ch
                channels.append(ch)
                
                # Add attention at specified resolutions
                if image_size // ds in attention_resolutions:
                    self.encoder_attns.append(AttentionBlock(ch, num_heads))
                else:
                    self.encoder_attns.append(nn.Identity())
            
            # Downsample (except last level)
            if level < len(channel_mult) - 1:
                self.encoder_blocks.append(
                    ResBlock(ch, ch, time_emb_dim, dropout, down=True)
                )
                channels.append(ch)
                self.encoder_attns.append(nn.Identity())
                ds *= 2
        
        # Middle block
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch, num_heads)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout)
        
        # 记录 mid feature channels 用于 hypergraph
        self.mid_channels = ch
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                self.decoder_blocks.append(
                    ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout)
                )
                ch = out_ch
                
                if image_size // ds in attention_resolutions:
                    self.decoder_attns.append(AttentionBlock(ch, num_heads))
                else:
                    self.decoder_attns.append(nn.Identity())
            
            # Upsample (except first level in reverse)
            if level > 0:
                self.decoder_blocks.append(
                    ResBlock(ch, ch, time_emb_dim, dropout, up=True)
                )
                self.decoder_attns.append(nn.Identity())
                ds //= 2
        
        # Output
        self.out_norm = GroupNorm32(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Zero-init output conv
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        cond: torch.Tensor,
        domain_id: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: (B, C, H, W) noisy image
            t: (B,) timesteps
            cond: (B, C, H, W) condition image
            domain_id: (B,) optional domain id for embedding
        
        Returns:
            noise_pred: (B, C, H, W)
            mid_features: (B, C_mid, H_mid, W_mid) if return_mid_features else None
        """
        # Time embedding
        t_emb = get_timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # Add domain embedding if provided
        if domain_id is not None:
            d_emb = self.domain_embed(domain_id)
            t_emb = t_emb + self.domain_proj(d_emb)
        
        # Concat input with condition
        h = torch.cat([x, cond], dim=1)
        h = self.input_conv(h)
        
        # Encoder
        skips = [h]
        block_idx = 0
        for block, attn in zip(self.encoder_blocks, self.encoder_attns):
            h = block(h, t_emb)
            h = attn(h)
            skips.append(h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        
        # 保存中间特征用于 hypergraph
        mid_features = h.clone() if self.return_mid_features else None
        
        h = self.mid_block2(h, t_emb)
        
        # Decoder
        for block, attn in zip(self.decoder_blocks, self.decoder_attns):
            if skips:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
            h = attn(h)
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h, mid_features
    
    def get_mid_feature_shape(self, input_size: int = 256) -> Tuple[int, int, int]:
        """
        获取中间特征的 shape
        
        Returns:
            (C, H, W)
        """
        num_downsamples = len([m for m in self.encoder_blocks if hasattr(m, 'down') and m.down])
        spatial = input_size // (2 ** num_downsamples)
        return (self.mid_channels, spatial, spatial)


# 测试代码
if __name__ == "__main__":
    # 测试 U-Net
    model = ConditionalUNet(
        in_channels=3,
        out_channels=3,
        model_channels=64,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(16,),
        image_size=128
    )
    
    B = 2
    x = torch.randn(B, 3, 128, 128)
    t = torch.randint(0, 1000, (B,))
    cond = torch.randn(B, 3, 128, 128)
    domain_id = torch.zeros(B, dtype=torch.long)
    
    noise_pred, mid_feat = model(x, t, cond, domain_id)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {noise_pred.shape}")
    print(f"Mid feature shape: {mid_feat.shape}")
    print(f"Expected mid shape: {model.get_mid_feature_shape(128)}")
    
    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

