"""
HGD 高性能训练脚本
针对 RTX PRO 6000 96GB 优化

主要优化:
1. 混合精度训练 (AMP)
2. 大 batch size
3. torch.compile() 加速
4. cuDNN benchmark
5. 梯度累积 (可选)
6. 高效数据加载
"""
import os
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from config import Config, get_default_config
from models import ConditionalUNet, GaussianDiffusion
from losses import HypergraphContrastiveLoss
from data import create_dataloader, UnpairedStainDataset, get_transforms


def setup_for_speed():
    """
    配置 PyTorch 以获得最佳性能
    """
    # cuDNN benchmark - 自动选择最快的卷积算法
    torch.backends.cudnn.benchmark = True
    
    # 允许 TF32 (Ampere+ GPU 支持)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 设置 float32 矩阵乘法精度为 'high' 或 'medium'
    torch.set_float32_matmul_precision('high')
    
    print("✓ Performance optimizations enabled:")
    print("  - cuDNN benchmark: ON")
    print("  - TF32: ON")
    print("  - Float32 matmul precision: high")


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FastHGDTrainer:
    """
    高性能 HGD 训练器
    
    针对大显存 GPU 优化:
    - 混合精度训练 (FP16/BF16)
    - 大 batch size
    - torch.compile() 模型编译
    - 高效内存管理
    """
    def __init__(
        self,
        config: Config,
        device: str = 'cuda',
        use_amp: bool = True,
        use_compile: bool = True,
        compile_mode: str = 'reduce-overhead'  # 'default', 'reduce-overhead', 'max-autotune'
    ):
        self.config = config
        self.device = torch.device(device)
        self.use_amp = use_amp
        
        # 检测 GPU 能力
        self._check_gpu()
        
        print(f"\n{'='*60}")
        print(f"FastHGDTrainer Configuration")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision (AMP): {use_amp}")
        print(f"torch.compile: {use_compile}")
        print(f"{'='*60}\n")
        
        # 创建模型
        self._build_model()
        
        # 编译模型 (PyTorch 2.0+)
        if use_compile and hasattr(torch, 'compile'):
            print(f"Compiling model with mode='{compile_mode}'...")
            self.unet = torch.compile(self.unet, mode=compile_mode)
            print("✓ Model compiled!")
        
        # 创建优化器
        self._build_optimizer()
        
        # 创建 loss
        self._build_loss()
        
        # 混合精度 scaler
        self.scaler = GradScaler() if use_amp else None
        
        # 确定 autocast dtype
        if use_amp:
            # BF16 对于 Ampere+ GPU 更好
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                print("Using BFloat16 for mixed precision")
            else:
                self.amp_dtype = torch.float16
                print("Using Float16 for mixed precision")
        
        # TensorBoard
        self.writer = None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
    
    def _check_gpu(self):
        """检查 GPU 信息"""
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"\n{'='*60}")
            print(f"GPU Information")
            print(f"{'='*60}")
            print(f"Name: {props.name}")
            print(f"Total Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"Compute Capability: {props.major}.{props.minor}")
            print(f"Multi Processors: {props.multi_processor_count}")
            
            # 建议 batch size
            memory_gb = props.total_memory / 1024**3
            if memory_gb >= 80:
                print(f"\n💡 Recommended batch_size for 256x256: 32-64")
                print(f"💡 Recommended batch_size for 512x512: 8-16")
            elif memory_gb >= 40:
                print(f"\n💡 Recommended batch_size for 256x256: 16-32")
            print(f"{'='*60}\n")
    
    def _build_model(self):
        """构建模型"""
        cfg = self.config
        
        # U-Net - 可以使用更大的模型
        self.unet = ConditionalUNet(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            model_channels=cfg.model.model_channels,
            channel_mult=cfg.model.channel_mult,
            num_res_blocks=cfg.model.num_res_blocks,
            attention_resolutions=cfg.model.attention_resolutions,
            dropout=cfg.model.dropout,
            num_domains=cfg.model.num_domains,
            domain_embed_dim=cfg.model.domain_embed_dim,
            image_size=cfg.model.image_size,
            return_mid_features=cfg.hypergraph.enabled
        ).to(self.device)
        
        # Diffusion
        self.diffusion = GaussianDiffusion(
            model=self.unet,
            num_timesteps=cfg.diffusion.num_timesteps,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
            beta_schedule=cfg.diffusion.beta_schedule,
            sampling_timesteps=cfg.diffusion.sampling_timesteps,
            ddim_eta=cfg.diffusion.ddim_eta
        ).to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.unet.parameters())
        print(f"U-Net parameters: {total_params / 1e6:.2f}M")
    
    def _build_optimizer(self):
        """构建优化器"""
        cfg = self.config.training
        
        # 使用 fused AdamW (更快)
        self.optimizer = AdamW(
            self.unet.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            fused=True  # 使用 CUDA fused 实现
        )
        
        # OneCycleLR 通常收敛更快
        # self.scheduler = OneCycleLR(
        #     self.optimizer,
        #     max_lr=cfg.learning_rate * 10,
        #     total_steps=cfg.num_epochs * 1000,  # 估计
        #     pct_start=0.1
        # )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.num_epochs,
            eta_min=cfg.learning_rate * 0.01
        )
    
    def _build_loss(self):
        """构建损失函数"""
        cfg = self.config
        
        self.hypergraph_loss = None
        if cfg.hypergraph.enabled:
            mid_channels, _, _ = self.unet.get_mid_feature_shape(cfg.model.image_size)
            
            self.hypergraph_loss = HypergraphContrastiveLoss(
                feature_dim=mid_channels,
                hidden_dim=cfg.hypergraph.hgnn_hidden_dim,
                output_dim=cfg.hypergraph.hgnn_hidden_dim,
                num_clusters=cfg.hypergraph.num_clusters,
                num_hgnn_layers=2,
                temperature=cfg.hypergraph.temperature,
                use_weighted_loss=True,
                membership_threshold=cfg.hypergraph.membership_threshold
            ).to(self.device)
            
            self.optimizer.add_param_group({
                'params': self.hypergraph_loss.parameters(),
                'lr': cfg.training.learning_rate
            })
    
    def train_step(self, batch: dict) -> dict:
        """
        单步训练 (带 AMP)
        """
        self.unet.train()
        
        source = batch['source'].to(self.device, non_blocking=True)
        target = batch['target'].to(self.device, non_blocking=True)
        B = source.shape[0]
        domain_id = torch.ones(B, dtype=torch.long, device=self.device)
        
        # 混合精度上下文
        amp_context = autocast(dtype=self.amp_dtype) if self.use_amp else nullcontext()
        
        with amp_context:
            # Forward diffusion + noise prediction
            diffusion_loss, noise_pred, mid_features_target = self.diffusion(
                x_0=target,
                cond=source,
                domain_id=domain_id
            )
            
            total_loss = self.config.training.lambda_diffusion * diffusion_loss
            
            losses = {'diffusion': diffusion_loss.item()}
            
            # Hypergraph loss
            if self.hypergraph_loss is not None and mid_features_target is not None:
                with torch.no_grad():
                    t_dummy = torch.zeros(B, dtype=torch.long, device=self.device)
                    _, mid_features_source = self.unet(source, t_dummy, source, domain_id=None)
                
                hg_loss, hg_info = self.hypergraph_loss(
                    mid_features_source.detach(),
                    mid_features_target
                )
                
                total_loss = total_loss + self.config.training.lambda_hypergraph * hg_loss
                losses['hypergraph'] = hg_loss.item()
            
            losses['total'] = total_loss.item()
        
        # Backward with AMP
        self.optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
        
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        return losses
    
    def train_epoch(self, dataloader, epoch: int) -> dict:
        """训练一个 epoch"""
        epoch_losses = {}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        for batch in pbar:
            losses = self.train_step(batch)
            
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{losses['total']:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            if self.writer is not None:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
            
            self.global_step += 1
        
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    def train(
        self,
        train_dataloader,
        num_epochs: int = None,
        save_dir: str = None
    ):
        """完整训练流程"""
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
        
        if save_dir is None:
            save_dir = self.config.checkpoint_dir
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        log_dir = save_dir / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir)
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {train_dataloader.batch_size}")
        print(f"Total batches per epoch: {len(train_dataloader)}")
        print(f"Checkpoints: {save_dir}")
        print(f"{'='*60}\n")
        
        best_loss = float('inf')
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            avg_losses = self.train_epoch(train_dataloader, epoch)
            
            # Log
            loss_str = ", ".join([f"{k}={v:.4f}" for k, v in avg_losses.items()])
            print(f"Epoch {epoch}: {loss_str}")
            
            self.scheduler.step()
            
            # Save
            if (epoch + 1) % self.config.training.save_interval == 0:
                self._save_checkpoint(save_dir / f'checkpoint_epoch_{epoch}.pt', epoch, avg_losses['total'])
            
            if avg_losses['total'] < best_loss:
                best_loss = avg_losses['total']
                self._save_checkpoint(save_dir / 'checkpoint_best.pt', epoch, best_loss)
        
        self._save_checkpoint(save_dir / 'checkpoint_final.pt', num_epochs - 1, avg_losses['total'])
        
        if self.writer:
            self.writer.close()
        
        print("\n✓ Training completed!")
    
    def _save_checkpoint(self, path, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)


def create_fast_dataloader(
    data_root: str,
    domain_a: str,
    domain_b: str,
    batch_size: int,
    image_size: int,
    num_workers: int = 8,
    prefetch_factor: int = 4
) -> DataLoader:
    """
    创建高性能 DataLoader
    """
    transform = get_transforms(image_size, is_train=True)
    
    dataset = UnpairedStainDataset(
        data_root=data_root,
        domain_a=domain_a,
        domain_b=domain_b,
        transform=transform,
        image_size=image_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='HGD Fast Training (for large GPU)')
    
    # Data
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--domain_a', type=str, default='HE')
    parser.add_argument('--domain_b', type=str, default='IHC')
    
    # Model - 大 GPU 可以用更大的模型
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--model_channels', type=int, default=192,  # 增大基础通道
                        help='Base model channels (default 192 for large GPU)')
    
    # Training - 大 batch size
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (32-64 for 96GB @ 256x256)')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate (can be higher with large batch)')
    
    # Hypergraph
    parser.add_argument('--use_hypergraph', action='store_true')
    parser.add_argument('--lambda_hg', type=float, default=0.1)
    parser.add_argument('--num_clusters', type=int, default=9)
    
    # Performance
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--no_compile', action='store_true', help='Disable torch.compile')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'])
    parser.add_argument('--num_workers', type=int, default=8)
    
    # Misc
    parser.add_argument('--output_dir', type=str, default='./outputs_fast')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 性能优化设置
    setup_for_speed()
    set_seed(args.seed)
    
    # 配置
    config = get_default_config()
    config.data_root = args.data_root
    config.domain_a = args.domain_a
    config.domain_b = args.domain_b
    config.output_dir = args.output_dir
    config.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    
    config.model.image_size = args.image_size
    config.model.model_channels = args.model_channels
    
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.num_epochs
    config.training.learning_rate = args.lr
    config.training.num_workers = args.num_workers
    config.training.lambda_hypergraph = args.lambda_hg
    
    config.hypergraph.enabled = args.use_hypergraph
    config.hypergraph.num_clusters = args.num_clusters
    
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # DataLoader
    train_dataloader = create_fast_dataloader(
        data_root=config.data_root,
        domain_a=config.domain_a,
        domain_b=config.domain_b,
        batch_size=config.training.batch_size,
        image_size=config.model.image_size,
        num_workers=config.training.num_workers
    )
    
    # Trainer
    trainer = FastHGDTrainer(
        config,
        use_amp=not args.no_amp,
        use_compile=not args.no_compile,
        compile_mode=args.compile_mode
    )
    
    # Train
    trainer.train(train_dataloader)


if __name__ == '__main__':
    main()

