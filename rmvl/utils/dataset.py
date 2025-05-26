import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class WatermarkDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        self.watermarked_dir = os.path.join(root_dir, 'watermarked')
        self.clean_dir = os.path.join(root_dir, 'clean')
        
        self.filenames = sorted(os.listdir(self.watermarked_dir))
        
        # 过滤无效文件
        self.filenames = [f for f in self.filenames if 
                          f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # 加载图像
        wm_path = os.path.join(self.watermarked_dir, filename)
        clean_path = os.path.join(self.clean_dir, filename)
        
        wm_image = Image.open(wm_path).convert('RGB')
        clean_image = Image.open(clean_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            wm_image = self.transform(wm_image)
            clean_image = self.transform(clean_image)
        
        # 生成水印掩码（如果不存在）
        mask = self._generate_mask(wm_image, clean_image)
        
        return {
            'watermarked': wm_image,
            'clean': clean_image,
            'mask': mask,
            'filename': filename
        }
    
    def _generate_mask(self, wm_image, clean_image, threshold=0.1):
        """生成水印区域的二值掩码"""
        diff = torch.abs(wm_image - clean_image).mean(dim=0)
        mask = (diff > threshold).float()
        return mask.unsqueeze(0)  # 添加通道维度