import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# 移除旧的配置导入，使用函数参数传递配置

class WatermarkDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', split_ratio=0.8, config=None):
        all_files = sorted([f for f in os.listdir(root_dir) if f.endswith('_watermark.png')])
        random.shuffle(all_files)
        
        split_idx = int(len(all_files) * split_ratio)
        
        if mode == 'train':
            self.files = all_files[:split_idx]
        else:
            self.files = all_files[split_idx:]
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.config = config
        
        self.watermarked_dir = os.path.join(root_dir, 'watermarked')
        self.clean_dir = os.path.join(root_dir, 'clean')
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        
        # 加载水印图像
        watermarked_path = os.path.join(self.watermarked_dir, filename)
        watermarked_image = Image.open(watermarked_path).convert('RGB')
        
        # 加载干净图像
        clean_filename = filename.replace('_watermark', '_clean')
        clean_path = os.path.join(self.clean_dir, clean_filename)
        clean_image = Image.open(clean_path).convert('RGB')
        
        # 生成掩码
        watermarked_np = np.array(watermarked_image)
        clean_np = np.array(clean_image)
        
        # 计算差异
        diff = np.abs(watermarked_np.astype(np.float32) - clean_np.astype(np.float32))
        diff_gray = np.mean(diff, axis=2)
        
        # 创建二值掩码
        threshold = 30
        mask = (diff_gray > threshold).astype(np.uint8) * 255
        mask = Image.fromarray(mask, mode='L')
        
        # 调整大小
        if self.config and 'data' in self.config and 'image_size' in self.config['data']:
            target_size = tuple(self.config['data']['image_size'])
            watermarked_image = watermarked_image.resize(target_size, Image.LANCZOS)
            clean_image = clean_image.resize(target_size, Image.LANCZOS)
            mask = mask.resize(target_size, Image.NEAREST)
        
        if self.transform:
            watermarked_image = self.transform(watermarked_image)
            clean_image = self.transform(clean_image)
            mask = self.transform(mask)
        
        return {
            'watermarked': watermarked_image,
            'clean': clean_image,
            'mask': mask,
            'filename': filename
        }

class SegWatermarkDataset(Dataset):
    def __init__(self, watermarked_dir, clean_dir, transform=None, augment=False):
        self.watermarked_dir = watermarked_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.augment = augment
        self.file_pairs = self._match_files()
        
    def _match_files(self):
        w_files = sorted([f for f in os.listdir(self.watermarked_dir) if f.endswith(('.png', '.jpg'))])
        c_files = sorted([f for f in os.listdir(self.clean_dir) if f.endswith(('.png', '.jpg'))])
        return list(zip(w_files, c_files))
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        w_file, c_file = self.file_pairs[idx]
        
        # 加载图像
        watermarked_path = os.path.join(self.watermarked_dir, w_file)
        clean_path = os.path.join(self.clean_dir, c_file)
        
        watermarked_image = Image.open(watermarked_path).convert('RGB')
        clean_image = Image.open(clean_path).convert('RGB')
        
        # 生成掩码
        watermarked_np = np.array(watermarked_image)
        clean_np = np.array(clean_image)
        
        diff = np.abs(watermarked_np.astype(np.float32) - clean_np.astype(np.float32))
        diff_gray = np.mean(diff, axis=2)
        
        threshold = 30
        mask = (diff_gray > threshold).astype(np.uint8) * 255
        mask = Image.fromarray(mask, mode='L')
        
        if self.transform:
            watermarked_image = self.transform(watermarked_image)
            clean_image = self.transform(clean_image)
            mask = self.transform(mask)
        
        return {
            'watermarked': watermarked_image,
            'clean': clean_image,
            'mask': mask
        }

def create_dataloaders(config):
    """创建数据加载器，接收配置参数"""
    transform = transforms.Compose([
        transforms.Resize(config['data']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = WatermarkDataset(
        config['data']['watermarked_dir'],
        config['data']['clean_dir'],
        transform=transform,
        config=config
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader
