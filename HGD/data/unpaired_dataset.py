"""
Unpaired Dataset for Virtual Stain Transfer

支持:
- 两个不配对的 domain (A, B)
- 随机采样配对
- 数据增强
"""
import os
import random
from pathlib import Path
from typing import Tuple, Optional, Callable, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def get_transforms(
    image_size: int = 256,
    is_train: bool = True,
    normalize: bool = True
) -> T.Compose:
    """
    获取数据变换
    
    Args:
        image_size: 输出图像大小
        is_train: 是否训练模式 (启用数据增强)
        normalize: 是否归一化到 [-1, 1]
    
    Returns:
        transform: torchvision transform
    """
    transforms = []
    
    # Resize
    transforms.append(T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR))
    transforms.append(T.CenterCrop(image_size))
    
    # 训练时的数据增强
    if is_train:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
        transforms.append(T.RandomVerticalFlip(p=0.5))
        # 可选: 颜色增强 (病理图像要谨慎使用)
        # transforms.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
    
    # To tensor
    transforms.append(T.ToTensor())
    
    # Normalize to [-1, 1] (diffusion 标准)
    if normalize:
        transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    
    return T.Compose(transforms)


class UnpairedStainDataset(Dataset):
    """
    Unpaired 数据集
    
    数据结构:
    data_root/
    ├── domain_a/  (e.g., HE)
    │   ├── img1.png
    │   ├── img2.png
    │   └── ...
    └── domain_b/  (e.g., IHC)
        ├── img1.png
        ├── img2.png
        └── ...
    
    注意: domain_a 和 domain_b 的图像不需要配对
    每次 __getitem__ 随机从两个 domain 各取一张图
    """
    def __init__(
        self,
        data_root: str,
        domain_a: str = "HE",
        domain_b: str = "IHC",
        transform: Optional[Callable] = None,
        image_size: int = 256,
        extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    ):
        """
        Args:
            data_root: 数据根目录
            domain_a: source domain 名称
            domain_b: target domain 名称
            transform: 数据变换 (如果 None 则使用默认)
            image_size: 图像大小
            extensions: 支持的图像格式
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.domain_a = domain_a
        self.domain_b = domain_b
        
        # 设置 transform
        if transform is None:
            self.transform = get_transforms(image_size, is_train=True)
        else:
            self.transform = transform
        
        # 加载图像路径
        self.images_a = self._load_images(self.data_root / domain_a, extensions)
        self.images_b = self._load_images(self.data_root / domain_b, extensions)
        
        if len(self.images_a) == 0:
            raise ValueError(f"No images found in {self.data_root / domain_a}")
        if len(self.images_b) == 0:
            raise ValueError(f"No images found in {self.data_root / domain_b}")
        
        print(f"Loaded {len(self.images_a)} images from domain A ({domain_a})")
        print(f"Loaded {len(self.images_b)} images from domain B ({domain_b})")
    
    def _load_images(self, folder: Path, extensions: Tuple[str, ...]) -> List[Path]:
        """加载文件夹中的所有图像路径"""
        if not folder.exists():
            return []
        
        images = []
        for ext in extensions:
            images.extend(folder.glob(f"*{ext}"))
            images.extend(folder.glob(f"*{ext.upper()}"))
        
        return sorted(images)
    
    def __len__(self) -> int:
        # 使用较大的 domain 作为 epoch 长度
        return max(len(self.images_a), len(self.images_b))
    
    def __getitem__(self, idx: int) -> dict:
        """
        获取一对图像 (随机配对)
        
        Returns:
            dict with:
                - 'source': (C, H, W) tensor from domain A
                - 'target': (C, H, W) tensor from domain B
                - 'source_path': str
                - 'target_path': str
                - 'domain_id_source': int (0)
                - 'domain_id_target': int (1)
        """
        # 随机选择图像 (unpaired)
        idx_a = idx % len(self.images_a)
        idx_b = random.randint(0, len(self.images_b) - 1)
        
        # 加载图像
        img_a = Image.open(self.images_a[idx_a]).convert('RGB')
        img_b = Image.open(self.images_b[idx_b]).convert('RGB')
        
        # 应用变换
        img_a = self.transform(img_a)
        img_b = self.transform(img_b)
        
        return {
            'source': img_a,
            'target': img_b,
            'source_path': str(self.images_a[idx_a]),
            'target_path': str(self.images_b[idx_b]),
            'domain_id_source': 0,
            'domain_id_target': 1
        }


class PairedStainDataset(Dataset):
    """
    Paired 数据集 (用于有配对数据的情况)
    
    数据结构:
    data_root/
    ├── domain_a/
    │   ├── sample_001.png
    │   └── ...
    └── domain_b/
        ├── sample_001.png  (与 domain_a 同名 = 配对)
        └── ...
    """
    def __init__(
        self,
        data_root: str,
        domain_a: str = "HE",
        domain_b: str = "IHC",
        transform: Optional[Callable] = None,
        image_size: int = 256,
        extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    ):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.domain_a = domain_a
        self.domain_b = domain_b
        
        if transform is None:
            self.transform = get_transforms(image_size, is_train=True)
        else:
            self.transform = transform
        
        # 加载配对图像
        self.pairs = self._load_paired_images(extensions)
        
        if len(self.pairs) == 0:
            raise ValueError(f"No paired images found")
        
        print(f"Loaded {len(self.pairs)} paired images")
    
    def _load_paired_images(self, extensions: Tuple[str, ...]) -> List[Tuple[Path, Path]]:
        """加载配对图像"""
        folder_a = self.data_root / self.domain_a
        folder_b = self.data_root / self.domain_b
        
        pairs = []
        
        for img_a in folder_a.iterdir():
            if img_a.suffix.lower() not in extensions:
                continue
            
            # 查找对应的 domain_b 图像
            for ext in extensions:
                img_b = folder_b / (img_a.stem + ext)
                if img_b.exists():
                    pairs.append((img_a, img_b))
                    break
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> dict:
        img_a_path, img_b_path = self.pairs[idx]
        
        img_a = Image.open(img_a_path).convert('RGB')
        img_b = Image.open(img_b_path).convert('RGB')
        
        # 同步数据增强 (对于配对数据很重要)
        seed = random.randint(0, 2**32 - 1)
        
        random.seed(seed)
        torch.manual_seed(seed)
        img_a = self.transform(img_a)
        
        random.seed(seed)
        torch.manual_seed(seed)
        img_b = self.transform(img_b)
        
        return {
            'source': img_a,
            'target': img_b,
            'source_path': str(img_a_path),
            'target_path': str(img_b_path),
            'domain_id_source': 0,
            'domain_id_target': 1
        }


def create_dataloader(
    data_root: str,
    domain_a: str = "HE",
    domain_b: str = "IHC",
    batch_size: int = 4,
    image_size: int = 256,
    num_workers: int = 4,
    is_train: bool = True,
    paired: bool = False
) -> DataLoader:
    """
    创建 DataLoader
    
    Args:
        data_root: 数据根目录
        domain_a: source domain
        domain_b: target domain
        batch_size: batch size
        image_size: 图像大小
        num_workers: 数据加载线程数
        is_train: 是否训练模式
        paired: 是否使用配对数据集
    
    Returns:
        DataLoader
    """
    transform = get_transforms(image_size, is_train=is_train)
    
    if paired:
        dataset = PairedStainDataset(
            data_root=data_root,
            domain_a=domain_a,
            domain_b=domain_b,
            transform=transform,
            image_size=image_size
        )
    else:
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
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )
    
    return dataloader


# 测试/演示
if __name__ == "__main__":
    print("Dataset module loaded.")
    print("Usage:")
    print("  from data import UnpairedStainDataset, create_dataloader")
    print("  dataloader = create_dataloader('./data', 'HE', 'IHC', batch_size=4)")

