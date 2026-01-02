"""
HGD 配置文件
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """模型配置"""
    # U-Net 配置
    image_size: int = 256
    in_channels: int = 3
    out_channels: int = 3
    model_channels: int = 128  # 基础通道数
    channel_mult: tuple = (1, 2, 4, 8)  # 各层通道倍数
    num_res_blocks: int = 2
    attention_resolutions: tuple = (16, 8)  # 在哪些分辨率使用 attention
    dropout: float = 0.0
    
    # Condition 配置
    condition_channels: int = 3  # source image 通道数
    num_domains: int = 2  # domain 数量 (用于 domain embedding)
    domain_embed_dim: int = 128


@dataclass
class DiffusionConfig:
    """扩散配置"""
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # linear, cosine
    
    # 采样配置
    sampling_timesteps: int = 50  # DDIM 采样步数
    ddim_eta: float = 0.0  # 0 = deterministic DDIM


@dataclass
class HypergraphConfig:
    """Hypergraph 配置"""
    enabled: bool = True
    num_clusters: int = 9  # hyperedge 数量
    feature_layer: str = "mid"  # 从哪层提取特征: "mid", "bottleneck"
    feature_dim: int = 512  # patch feature 维度
    hgnn_hidden_dim: int = 256
    membership_threshold: float = 0.15  # soft clustering 阈值
    temperature: float = 0.07  # contrastive loss 温度


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    num_epochs: int = 100
    save_interval: int = 10
    log_interval: int = 100
    
    # Loss 权重
    lambda_diffusion: float = 1.0
    lambda_hypergraph: float = 0.1
    
    # 数据
    num_workers: int = 4
    image_size: int = 256


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    hypergraph: HypergraphConfig = field(default_factory=HypergraphConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 路径
    data_root: str = "./data"
    domain_a: str = "HE"
    domain_b: str = "IHC"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    
    # 设备
    device: str = "cuda"
    seed: int = 42


def get_default_config() -> Config:
    """获取默认配置"""
    return Config()

