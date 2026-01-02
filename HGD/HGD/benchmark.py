"""
性能基准测试脚本
用于测试不同配置下的训练速度和显存占用
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import time
from contextlib import nullcontext

from models import ConditionalUNet, GaussianDiffusion
from losses import HypergraphContrastiveLoss


def benchmark_config(
    image_size: int,
    model_channels: int,
    batch_size: int,
    use_amp: bool = True,
    num_iterations: int = 50,
    warmup: int = 10
):
    """
    测试特定配置的性能
    """
    device = torch.device('cuda')
    
    # 清空缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 创建模型
    unet = ConditionalUNet(
        in_channels=3,
        out_channels=3,
        model_channels=model_channels,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        image_size=image_size,
        return_mid_features=True
    ).to(device)
    
    diffusion = GaussianDiffusion(
        model=unet,
        num_timesteps=1000
    ).to(device)
    
    # 尝试编译
    try:
        unet = torch.compile(unet, mode='reduce-overhead')
    except Exception as e:
        print(f"  torch.compile not available: {e}")
    
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    
    # 创建测试数据
    source = torch.randn(batch_size, 3, image_size, image_size, device=device)
    target = torch.randn(batch_size, 3, image_size, image_size, device=device)
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        with autocast(dtype=amp_dtype) if use_amp else nullcontext():
            loss, _, _ = diffusion(target, source)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"  Benchmarking ({num_iterations} iterations)...")
    start_time = time.time()
    
    for _ in range(num_iterations):
        with autocast(dtype=amp_dtype) if use_amp else nullcontext():
            loss, _, _ = diffusion(target, source)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 统计
    total_time = end_time - start_time
    time_per_iter = total_time / num_iterations * 1000  # ms
    images_per_sec = batch_size * num_iterations / total_time
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    # 参数量
    params = sum(p.numel() for p in unet.parameters()) / 1e6
    
    return {
        'time_per_iter_ms': time_per_iter,
        'images_per_sec': images_per_sec,
        'peak_memory_gb': peak_memory,
        'params_m': params
    }


def run_benchmark():
    """运行完整基准测试"""
    print("=" * 70)
    print("HGD Performance Benchmark")
    print("=" * 70)
    
    # 检查 GPU
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"Memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"Compute: {props.major}.{props.minor}")
    
    # 性能优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 测试配置
    configs = [
        # (image_size, model_channels, batch_size, description)
        (256, 128, 8, "256x256, ch=128, bs=8 (baseline)"),
        (256, 128, 16, "256x256, ch=128, bs=16"),
        (256, 128, 32, "256x256, ch=128, bs=32"),
        (256, 128, 48, "256x256, ch=128, bs=48"),
        (256, 192, 32, "256x256, ch=192, bs=32 (larger model)"),
        (256, 256, 24, "256x256, ch=256, bs=24 (large model)"),
        (512, 128, 4, "512x512, ch=128, bs=4"),
        (512, 128, 8, "512x512, ch=128, bs=8"),
        (512, 128, 12, "512x512, ch=128, bs=12"),
    ]
    
    print("\n" + "-" * 70)
    print(f"{'Config':<40} | {'ms/iter':>8} | {'img/s':>8} | {'Mem GB':>8}")
    print("-" * 70)
    
    for img_size, channels, bs, desc in configs:
        try:
            results = benchmark_config(
                image_size=img_size,
                model_channels=channels,
                batch_size=bs,
                use_amp=True,
                num_iterations=30,
                warmup=5
            )
            
            print(f"{desc:<40} | {results['time_per_iter_ms']:>8.1f} | "
                  f"{results['images_per_sec']:>8.1f} | {results['peak_memory_gb']:>8.1f}")
            
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{desc:<40} | {'OOM':>8} | {'-':>8} | {'-':>8}")
                torch.cuda.empty_cache()
            else:
                raise e
    
    print("-" * 70)
    print("\n✓ Benchmark complete!")
    print("\n建议: 选择 img/s 最高且 Mem GB < 90 的配置")


if __name__ == "__main__":
    run_benchmark()

