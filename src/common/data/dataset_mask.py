import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from PIL import Image, ImageFile


class WatermarkDataset(Dataset):
    def __init__(self, watermarked_dir, clean_dir=None, mask_dir=None, 
                 transform=None, mode='train', generate_mask_threshold=30):
        self.watermarked_dir = watermarked_dir
        self.clean_dir = clean_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mode = mode
        self.generate_mask_threshold = generate_mask_threshold
        
        # 检查目录是否存在
        if not os.path.exists(watermarked_dir):
            raise ValueError(f"Watermarked directory {watermarked_dir} does not exist!")
        
        # 收集所有图像文件名
        self.image_files = sorted([f for f in os.listdir(watermarked_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # 验证目录结构是否匹配
        if self.mode == 'train':
            if clean_dir is None:
                if mask_dir is None:
                    raise ValueError("For training mode, either clean_dir or mask_dir must be provided!")
            elif not os.path.exists(clean_dir):
                raise ValueError(f"Clean directory {clean_dir} does not exist!")
        
        # 创建掩码目录（如果不存在）
        if self.mask_dir and not os.path.exists(self.mask_dir):
            os.makedirs(self.mask_dir, exist_ok=True)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        watermarked_path = os.path.join(self.watermarked_dir, image_name)
        
        # 读取带水印图像
        watermarked_img = cv2.imread(watermarked_path)
        if watermarked_img is None:
            raise ValueError(f"Failed to read image: {watermarked_path}")
        watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB)
        
        # 训练模式需要掩码
        if self.mode == 'train':
            # 如果提供了掩码目录，直接读取
            if self.mask_dir and os.path.exists(os.path.join(self.mask_dir, image_name)):
                mask_path = os.path.join(self.mask_dir, image_name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Failed to read mask: {mask_path}")
            
            # 否则从带水印图像和无水印图像生成掩码
            else:
                if self.clean_dir is None:
                    raise ValueError("Mask not found and clean_dir not provided!")
                
                clean_path = os.path.join(self.clean_dir, image_name)
                clean_img = cv2.imread(clean_path)
                if clean_img is None:
                    raise ValueError(f"Failed to read clean image: {clean_path}")
                clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
                
                # 确保图像尺寸一致
                if watermarked_img.shape != clean_img.shape:
                    clean_img = cv2.resize(clean_img, (watermarked_img.shape[1], watermarked_img.shape[0]))
                
                # 生成掩码
                mask = self._generate_mask(watermarked_img, clean_img)
                
                # 保存生成的掩码（可选）
                if self.mask_dir:
                    mask_path = os.path.join(self.mask_dir, image_name)
                    cv2.imwrite(mask_path, mask)
            
            # 应用数据增强
            if self.transform:
                augmented = self.transform(image=watermarked_img, mask=mask)
                watermarked_img = augmented['image']
                mask = augmented['mask']
            
            # 确保掩码是二值的 - 兼容NumPy和Tensor
            if isinstance(mask, torch.Tensor):
                mask = (mask > 0.5).float()
            else:
                mask = (mask > 0.5).astype(np.float32)
            
            return watermarked_img, mask
        
        # 测试模式只返回带水印图像
        else:
            if self.transform:
                augmented = self.transform(image=watermarked_img)
                watermarked_img = augmented['image']
            
            return watermarked_img
    
    def _generate_mask(self, watermarked_img, clean_img):
        """从带水印图像和无水印图像生成掩码"""
        # 转换为灰度图
        gray_watermarked = cv2.cvtColor(watermarked_img, cv2.COLOR_RGB2GRAY)
        gray_clean = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)
        
        # 计算绝对差异
        diff = cv2.absdiff(gray_watermarked, gray_clean)
        
        # 二值化
        _, mask = cv2.threshold(diff, self.generate_mask_threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作优化掩码
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return mask

def get_train_transform(img_size=512):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.GaussianBlur(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transform(img_size=512):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def create_datasets(cfg):
    """创建训练集和验证集，自动划分"""
    # 创建完整数据集
    full_dataset = WatermarkDataset(
        watermarked_dir=os.path.join(cfg.DATA.ROOT_DIR, "watermarked"),
        clean_dir=os.path.join(cfg.DATA.ROOT_DIR, "clean"),
        mask_dir=os.path.join(cfg.DATA.ROOT_DIR, "masks"),
        transform=get_train_transform(cfg.DATA.IMG_SIZE),
        mode='train',
        generate_mask_threshold=cfg.DATA.GENERATE_MASK_THRESHOLD
    )
    
    # 设置随机种子确保可复现
    random.seed(cfg.DATA.SEED)
    torch.manual_seed(cfg.DATA.SEED)
    
    # 计算数据集划分
    dataset_size = len(full_dataset)
    train_size = int(cfg.DATA.TRAIN_RATIO * dataset_size)
    val_size = dataset_size - train_size
    
    # 随机划分数据集
    indices = list(range(dataset_size))
    if cfg.DATA.SHUFFLE:
        random.shuffle(indices)
    
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    
    # 创建训练集和验证集
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # 为验证集设置验证转换
    val_dataset.dataset.transform = get_val_transform(cfg.DATA.IMG_SIZE)
    
    print(f"数据集划分完成:")
    print(f"训练集: {len(train_dataset)} 张图像")
    print(f"验证集: {len(val_dataset)} 张图像")
    
    return train_dataset, val_dataset