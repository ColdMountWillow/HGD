"""
HGD Sampling / Inference Script
从训练好的模型生成虚拟染色图像

用法:
    python sample.py --checkpoint ./checkpoints/best.pt --input_dir ./test_images --output_dir ./results
"""
import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm

from config import get_default_config
from models import ConditionalUNet, GaussianDiffusion
from data import get_transforms


def denormalize(tensor):
    """从 [-1, 1] 反归一化到 [0, 1]"""
    return (tensor + 1) / 2


def load_model(checkpoint_path: str, config=None, device='cuda'):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: checkpoint 文件路径
        config: 配置对象 (如果 None 则使用默认)
        device: 设备
    
    Returns:
        diffusion: GaussianDiffusion 模型
    """
    if config is None:
        config = get_default_config()
    
    # 创建 U-Net
    unet = ConditionalUNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        model_channels=config.model.model_channels,
        channel_mult=config.model.channel_mult,
        num_res_blocks=config.model.num_res_blocks,
        attention_resolutions=config.model.attention_resolutions,
        dropout=0.0,  # 推理时不用 dropout
        num_domains=config.model.num_domains,
        domain_embed_dim=config.model.domain_embed_dim,
        image_size=config.model.image_size,
        return_mid_features=False  # 推理时不需要中间特征
    ).to(device)
    
    # 创建 Diffusion
    diffusion = GaussianDiffusion(
        model=unet,
        num_timesteps=config.diffusion.num_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_schedule=config.diffusion.beta_schedule,
        sampling_timesteps=config.diffusion.sampling_timesteps,
        ddim_eta=config.diffusion.ddim_eta
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Loss: {checkpoint.get('loss', 'unknown')}")
    
    unet.eval()
    
    return diffusion


@torch.no_grad()
def sample_single(
    diffusion,
    source_image: torch.Tensor,
    device: str = 'cuda',
    use_ddim: bool = True,
    show_progress: bool = False
) -> torch.Tensor:
    """
    对单张图像进行虚拟染色转换
    
    Args:
        diffusion: GaussianDiffusion 模型
        source_image: (1, C, H, W) 或 (C, H, W) source 图像
        device: 设备
        use_ddim: 是否使用 DDIM 采样
        show_progress: 是否显示进度
    
    Returns:
        generated: (1, C, H, W) 生成的图像
    """
    if source_image.dim() == 3:
        source_image = source_image.unsqueeze(0)
    
    source_image = source_image.to(device)
    
    # 定义进度回调
    def progress_callback(step, total):
        if show_progress:
            print(f"\rSampling: {step}/{total}", end="")
    
    # 生成
    generated = diffusion.sample(
        cond=source_image,
        domain_id=None,
        use_ddim=use_ddim,
        progress_callback=progress_callback if show_progress else None
    )
    
    if show_progress:
        print()
    
    return generated


@torch.no_grad()
def sample_batch(
    diffusion,
    source_images: torch.Tensor,
    device: str = 'cuda',
    use_ddim: bool = True
) -> torch.Tensor:
    """
    批量生成
    
    Args:
        diffusion: GaussianDiffusion 模型
        source_images: (B, C, H, W) source 图像
        device: 设备
        use_ddim: 是否使用 DDIM 采样
    
    Returns:
        generated: (B, C, H, W) 生成的图像
    """
    source_images = source_images.to(device)
    generated = diffusion.sample(cond=source_images, use_ddim=use_ddim)
    return generated


def process_directory(
    diffusion,
    input_dir: str,
    output_dir: str,
    image_size: int = 256,
    device: str = 'cuda',
    use_ddim: bool = True,
    save_comparison: bool = True
):
    """
    处理整个目录的图像
    
    Args:
        diffusion: GaussianDiffusion 模型
        input_dir: 输入目录
        output_dir: 输出目录
        image_size: 图像大小
        device: 设备
        use_ddim: 是否使用 DDIM
        save_comparison: 是否保存对比图
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建 transform
    transform = get_transforms(image_size, is_train=False, normalize=True)
    
    # 获取所有图像文件
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    for img_path in tqdm(image_files, desc="Generating"):
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        source = transform(img).unsqueeze(0)  # (1, C, H, W)
        
        # 生成
        generated = sample_single(diffusion, source, device, use_ddim)
        
        # 保存
        output_name = img_path.stem + '_generated.png'
        generated_denorm = denormalize(generated)
        save_image(generated_denorm, output_dir / output_name)
        
        # 保存对比图
        if save_comparison:
            source_denorm = denormalize(source)
            comparison = torch.cat([source_denorm, generated_denorm], dim=3)  # 水平拼接
            comparison_name = img_path.stem + '_comparison.png'
            save_image(comparison, output_dir / comparison_name)
    
    print(f"Results saved to {output_dir}")


def interactive_demo(
    diffusion,
    image_size: int = 256,
    device: str = 'cuda'
):
    """
    交互式演示
    """
    transform = get_transforms(image_size, is_train=False, normalize=True)
    
    print("\n=== HGD Interactive Demo ===")
    print("Enter image path to generate, or 'q' to quit.")
    
    while True:
        path = input("\nImage path: ").strip()
        
        if path.lower() == 'q':
            break
        
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        
        try:
            # 加载和处理
            img = Image.open(path).convert('RGB')
            source = transform(img).unsqueeze(0)
            
            print("Generating...")
            generated = sample_single(diffusion, source, device, use_ddim=True, show_progress=True)
            
            # 保存
            output_path = path.replace('.', '_generated.')
            generated_denorm = denormalize(generated)
            save_image(generated_denorm, output_path)
            print(f"Saved to: {output_path}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")


def parse_args():
    parser = argparse.ArgumentParser(description='HGD Sampling')
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    
    # Mode
    parser.add_argument('--input_dir', type=str, default=None, help='Input directory')
    parser.add_argument('--input_image', type=str, default=None, help='Single input image')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    # Options
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--model_channels', type=int, default=128, help='Model channels (must match training)')
    parser.add_argument('--sampling_steps', type=int, default=50, help='DDIM sampling steps')
    parser.add_argument('--no_ddim', action='store_true', help='Use DDPM instead of DDIM')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--no_comparison', action='store_true', help='Do not save comparison images')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建配置
    config = get_default_config()
    config.model.image_size = args.image_size
    config.model.model_channels = args.model_channels
    config.diffusion.sampling_timesteps = args.sampling_steps
    
    # 确定设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 加载模型
    diffusion = load_model(args.checkpoint, config, device)
    
    use_ddim = not args.no_ddim
    
    if args.interactive:
        # 交互模式
        interactive_demo(diffusion, args.image_size, device)
    
    elif args.input_image:
        # 单张图像
        transform = get_transforms(args.image_size, is_train=False)
        
        img = Image.open(args.input_image).convert('RGB')
        source = transform(img).unsqueeze(0)
        
        print("Generating...")
        generated = sample_single(diffusion, source, device, use_ddim, show_progress=True)
        
        # 保存
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'generated.png')
        save_image(denormalize(generated), output_path)
        print(f"Saved to: {output_path}")
        
        if not args.no_comparison:
            comparison = torch.cat([denormalize(source), denormalize(generated)], dim=3)
            comparison_path = os.path.join(args.output_dir, 'comparison.png')
            save_image(comparison, comparison_path)
            print(f"Comparison saved to: {comparison_path}")
    
    elif args.input_dir:
        # 目录处理
        process_directory(
            diffusion=diffusion,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            image_size=args.image_size,
            device=device,
            use_ddim=use_ddim,
            save_comparison=not args.no_comparison
        )
    
    else:
        print("Please specify --input_dir, --input_image, or --interactive")


if __name__ == '__main__':
    main()

