"""
无配对数据集加载器
用于加载 source 和 target domain 的图像
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, Optional
import glob


class UnpairedStainDataset(Dataset):
    """
    无配对虚拟染色数据集
    
    从两个不同的文件夹加载 source 和 target domain 的图像
    不需要像素级配对
    """
    
    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            source_dir: Source domain 图像文件夹路径
            target_dir: Target domain 图像文件夹路径
            image_size: 图像尺寸
            transform: 可选的额外变换
        """
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.image_size = image_size
        
        # 加载图像路径
        self.source_paths = sorted(glob.glob(os.path.join(source_dir, "*.png")) +
                                   glob.glob(os.path.join(source_dir, "*.jpg")) +
                                   glob.glob(os.path.join(source_dir, "*.jpeg")))
        self.target_paths = sorted(glob.glob(os.path.join(target_dir, "*.png")) +
                                   glob.glob(os.path.join(target_dir, "*.jpg")) +
                                   glob.glob(os.path.join(target_dir, "*.jpeg")))
        
        if len(self.source_paths) == 0:
            raise ValueError(f"No images found in {source_dir}")
        if len(self.target_paths) == 0:
            raise ValueError(f"No images found in {target_dir}")
        
        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return max(len(self.source_paths), len(self.target_paths))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回一对 source 和 target 图像（随机采样，无需配对）
        
        Args:
            idx: 索引
        
        Returns:
            source_image: Source domain 图像 [C, H, W]
            target_image: Target domain 图像 [C, H, W]
        """
        # 随机采样（无需配对）
        source_idx = idx % len(self.source_paths)
        target_idx = idx % len(self.target_paths)
        
        # 加载图像
        source_image = Image.open(self.source_paths[source_idx]).convert("RGB")
        target_image = Image.open(self.target_paths[target_idx]).convert("RGB")
        
        # 应用变换
        source_image = self.transform(source_image)
        target_image = self.transform(target_image)
        
        return source_image, target_image

