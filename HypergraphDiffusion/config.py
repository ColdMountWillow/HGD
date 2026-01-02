"""
配置文件：定义训练和模型参数
"""
import argparse
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """训练配置"""
    # 数据路径
    data_root: str = "./data"
    source_domain: str = "H&E"  # Source stain domain
    target_domain: str = "PAS"  # Target stain domain
    
    # 模型参数
    image_size: int = 256
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 64
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16, 8)
    
    # Diffusion 参数
    num_timesteps: int = 1000
    beta_schedule: str = "linear"  # "linear" or "cosine"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # Hypergraph 参数
    num_hyperedges: int = 9
    patch_size: int = 64  # Patch 采样数量（不是空间尺寸）
    hyperedge_threshold: float = 0.15
    feature_dim: int = 256  # 从 U-Net 提取的特征维度
    
    # 损失权重
    diffusion_loss_weight: float = 1.0
    hypergraph_loss_weight: float = 0.1
    temperature: float = 0.07  # Contrastive loss temperature
    
    # 训练参数
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    save_freq: int = 10
    sample_freq: int = 5
    num_workers: int = 4
    
    # 设备
    device: str = "cuda"
    seed: int = 42
    
    # 路径
    checkpoint_dir: str = "./checkpoints"
    sample_dir: str = "./samples"
    log_dir: str = "./logs"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Hypergraph-guided Diffusion Training")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(args=None) -> Config:
    """加载配置"""
    if args is None:
        args = parse_args()
    
    config = Config()
    
    # 从命令行参数覆盖配置
    if args.data_root is not None:
        config.data_root = args.data_root
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.device is not None:
        config.device = args.device
    
    return config

