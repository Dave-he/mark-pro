import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from configs.default import cfg

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
        w_path = os.path.join(self.watermarked_dir, w_file)
        c_path = os.path.join(self.clean_dir, c_file)
        w_img = Image.open(w_path).convert('RGB')
        c_img = Image.open(c_path).convert('RGB')
        # 先resize/crop/pad
        if cfg.data.resize_mode == 'fixed':
            w_img = w_img.resize(tuple(cfg.data.image_size), Image.BICUBIC)
            c_img = c_img.resize(tuple(cfg.data.image_size), Image.BICUBIC)
        elif cfg.data.resize_mode == 'scale':
            target_size = min(cfg.data.image_size)
            def resize_and_crop(img):
                w, h = img.size
                scale = target_size / min(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.BICUBIC)
                left = (new_w - target_size) // 2
                top = (new_h - target_size) // 2
                img = img.crop((left, top, left + target_size, top + target_size))
                return img
            w_img = resize_and_crop(w_img)
            c_img = resize_and_crop(c_img)
        elif cfg.data.resize_mode == 'pad':
            target_w, target_h = cfg.data.image_size
            def pad_to_size(img, fill=0):
                w, h = img.size
                pad_w = max(0, target_w - w)
                pad_h = max(0, target_h - h)
                padding = (pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2)
                return transforms.Pad(padding, fill=fill)(img)
            w_img = pad_to_size(w_img, fill=cfg.data.pad_value)
            c_img = pad_to_size(c_img, fill=cfg.data.pad_value)
        # 用resize/crop/pad后图片生成mask
        mask = self._create_mask(w_img, c_img)
        # mask同样resize/crop/pad（如果需要）
        mask = mask.resize(tuple(cfg.data.image_size), Image.NEAREST)
        # 转tensor并归一化
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        w_img = normalize(to_tensor(w_img))
        c_img = normalize(to_tensor(c_img))
        mask = to_tensor(mask)
        if self.augment:
            w_img, c_img, mask = self._augment(w_img, c_img, mask)
        return w_img, c_img, mask
    
    def _create_mask(self, w_img, c_img):
        w_np = np.array(w_img).astype(np.float32)
        c_np = np.array(c_img).astype(np.float32)
        diff = np.mean(np.abs(w_np - c_np), axis=2)
        mask = (diff > cfg.data.mask_threshold).astype(np.float32)
        return Image.fromarray(mask * 255).convert('L')
    
    def transform_mask(self, mask):
        mask = mask.resize(cfg.data.image_size, Image.NEAREST)
        return transforms.ToTensor()(mask)
    
    def _augment(self, w_img, c_img, mask):
        # 数据增强（随机翻转、旋转等）
        if np.random.random() > 0.5:
            w_img = transforms.functional.hflip(w_img)
            c_img = transforms.functional.hflip(c_img)
            mask = transforms.functional.hflip(mask)
        
        if np.random.random() > 0.5:
            w_img = transforms.functional.vflip(w_img)
            c_img = transforms.functional.vflip(c_img)
            mask = transforms.functional.vflip(mask)
        
        angle = np.random.uniform(-15, 15)
        w_img = transforms.functional.rotate(w_img, angle)
        c_img = transforms.functional.rotate(c_img, angle)
        mask = transforms.functional.rotate(mask, angle)
        
        return w_img, c_img, mask

def create_dataloaders():
    transform = transforms.Compose([
        transforms.Resize(cfg.data.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = SegWatermarkDataset(
        cfg.data.watermarked_dir,
        cfg.data.clean_dir,
        transform=transform,
        augment=True
    )
    
    train_size = int(cfg.train.validation_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader
