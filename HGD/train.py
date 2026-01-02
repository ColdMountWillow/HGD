"""
HGD Training Script
Hypergraph-guided Diffusion for Unpaired Virtual Stain Translation

用法:
    python train.py --data_root /path/to/data --domain_a HE --domain_b IHC
    python train.py --data_root /path/to/data --use_hypergraph --lambda_hg 0.1
"""
import os
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local imports
from config import Config, get_default_config
from models import ConditionalUNet, GaussianDiffusion
from losses import HypergraphContrastiveLoss
from data import create_dataloader


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str
):
    """保存 checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str
) -> int:
    """加载 checkpoint，返回 epoch"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


class HGDTrainer:
    """
    HGD 训练器
    
    训练流程:
    1. 从 domain A (source) 采样图像作为条件
    2. 从 domain B (target) 采样图像作为目标
    3. 前向扩散: 给 target 加噪
    4. 预测噪声并计算 diffusion loss
    5. (可选) 从中间特征计算 hypergraph loss
    6. 反向传播更新
    """
    def __init__(
        self,
        config: Config,
        device: str = 'cuda'
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # 创建模型
        self._build_model()
        
        # 创建优化器
        self._build_optimizer()
        
        # 创建 loss
        self._build_loss()
        
        # TensorBoard
        self.writer = None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
    
    def _build_model(self):
        """构建模型"""
        cfg = self.config
        
        # U-Net
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
        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        print(f"U-Net parameters: {total_params / 1e6:.2f}M (trainable: {trainable_params / 1e6:.2f}M)")
    
    def _build_optimizer(self):
        """构建优化器"""
        cfg = self.config.training
        
        self.optimizer = AdamW(
            self.unet.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.num_epochs,
            eta_min=cfg.learning_rate * 0.1
        )
    
    def _build_loss(self):
        """构建损失函数"""
        cfg = self.config
        
        # Hypergraph loss (可选)
        self.hypergraph_loss = None
        if cfg.hypergraph.enabled:
            # 获取 mid feature 维度
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
            
            # 将 hypergraph loss 的参数也加入优化器
            self.optimizer.add_param_group({
                'params': self.hypergraph_loss.parameters(),
                'lr': cfg.training.learning_rate
            })
            
            hg_params = sum(p.numel() for p in self.hypergraph_loss.parameters())
            print(f"Hypergraph loss parameters: {hg_params / 1e6:.2f}M")
    
    def train_step(self, batch: dict) -> dict:
        """
        单步训练
        
        Args:
            batch: dict with 'source', 'target', etc.
        
        Returns:
            losses: dict with loss values
        """
        self.unet.train()
        
        source = batch['source'].to(self.device)  # condition
        target = batch['target'].to(self.device)  # target (to generate)
        
        B = source.shape[0]
        
        # Domain IDs (可选)
        domain_id = torch.ones(B, dtype=torch.long, device=self.device)  # target domain = 1
        
        # Forward diffusion + noise prediction
        diffusion_loss, noise_pred, mid_features_target = self.diffusion(
            x_0=target,
            cond=source,
            domain_id=domain_id
        )
        
        total_loss = self.config.training.lambda_diffusion * diffusion_loss
        
        losses = {
            'diffusion': diffusion_loss.item(),
        }
        
        # Hypergraph loss (可选)
        if self.hypergraph_loss is not None and mid_features_target is not None:
            # 我们需要 source 的中间特征
            # 这里我们做一个简化: 直接用 source 过一遍 U-Net 的 encoder
            # 实际上可以设计更优雅的方式
            
            # 获取 source 的中间特征 (通过 dummy forward)
            with torch.no_grad():
                t_dummy = torch.zeros(B, dtype=torch.long, device=self.device)
                _, mid_features_source = self.unet(source, t_dummy, source, domain_id=None)
            
            # Hypergraph contrastive loss
            hg_loss, hg_info = self.hypergraph_loss(
                mid_features_source.detach(),  # source features (detach 避免梯度)
                mid_features_target  # target features (需要梯度)
            )
            
            total_loss = total_loss + self.config.training.lambda_hypergraph * hg_loss
            
            losses['hypergraph'] = hg_loss.item()
            losses['hg_loss_s2t'] = hg_info['loss_s2t']
            losses['hg_loss_t2s'] = hg_info['loss_t2s']
        
        losses['total'] = total_loss.item()
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return losses
    
    def train_epoch(self, dataloader, epoch: int) -> dict:
        """
        训练一个 epoch
        """
        epoch_losses = {}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            losses = self.train_step(batch)
            
            # 累积 losses
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v)
            
            # 更新进度条
            pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
            
            # TensorBoard logging
            if self.writer is not None:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
            
            self.global_step += 1
        
        # 计算平均
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self, dataloader, num_samples: int = 4) -> dict:
        """
        验证: 生成一些样本
        """
        self.unet.eval()
        
        # 取一个 batch
        batch = next(iter(dataloader))
        source = batch['source'][:num_samples].to(self.device)
        
        # 生成
        generated = self.diffusion.sample(source, use_ddim=True)
        
        return {
            'source': source.cpu(),
            'generated': generated.cpu()
        }
    
    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        num_epochs: int = None,
        save_dir: str = None
    ):
        """
        完整训练流程
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
        
        if save_dir is None:
            save_dir = self.config.checkpoint_dir
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        log_dir = save_dir / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir)
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Checkpoints will be saved to: {save_dir}")
        print(f"TensorBoard logs: {log_dir}")
        
        best_loss = float('inf')
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            avg_losses = self.train_epoch(train_dataloader, epoch)
            
            # Log
            print(f"Epoch {epoch}: " + ", ".join([f"{k}={v:.4f}" for k, v in avg_losses.items()]))
            
            # LR scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            if self.writer:
                self.writer.add_scalar('train/lr', current_lr, epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_interval == 0:
                save_checkpoint(
                    self.unet, self.optimizer, epoch, avg_losses['total'],
                    str(save_dir / f'checkpoint_epoch_{epoch}.pt')
                )
            
            # Save best
            if avg_losses['total'] < best_loss:
                best_loss = avg_losses['total']
                save_checkpoint(
                    self.unet, self.optimizer, epoch, best_loss,
                    str(save_dir / 'checkpoint_best.pt')
                )
            
            # Validation / sample generation
            if val_dataloader is not None and (epoch + 1) % self.config.training.save_interval == 0:
                samples = self.validate(val_dataloader)
                # 可以保存或可视化 samples
        
        # Final save
        save_checkpoint(
            self.unet, self.optimizer, num_epochs - 1, avg_losses['total'],
            str(save_dir / 'checkpoint_final.pt')
        )
        
        if self.writer:
            self.writer.close()
        
        print("Training completed!")


def parse_args():
    parser = argparse.ArgumentParser(description='HGD Training')
    
    # Data
    parser.add_argument('--data_root', type=str, required=True, help='Path to data root')
    parser.add_argument('--domain_a', type=str, default='HE', help='Source domain name')
    parser.add_argument('--domain_b', type=str, default='IHC', help='Target domain name')
    
    # Model
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--model_channels', type=int, default=128, help='Base model channels')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    # Hypergraph
    parser.add_argument('--use_hypergraph', action='store_true', help='Enable hypergraph loss')
    parser.add_argument('--lambda_hg', type=float, default=0.1, help='Hypergraph loss weight')
    parser.add_argument('--num_clusters', type=int, default=9, help='Number of hyperedges')
    
    # Misc
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create config
    config = get_default_config()
    
    # Override with args
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
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Create dataloader
    train_dataloader = create_dataloader(
        data_root=config.data_root,
        domain_a=config.domain_a,
        domain_b=config.domain_b,
        batch_size=config.training.batch_size,
        image_size=config.model.image_size,
        num_workers=config.training.num_workers,
        is_train=True,
        paired=False
    )
    
    # Create trainer
    trainer = HGDTrainer(config)
    
    # Resume if specified
    if args.resume:
        epoch = load_checkpoint(trainer.unet, trainer.optimizer, args.resume)
        trainer.current_epoch = epoch + 1
        print(f"Resumed from epoch {epoch}")
    
    # Train
    trainer.train(train_dataloader)


if __name__ == '__main__':
    main()

