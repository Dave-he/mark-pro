import os
import argparse
# 移除 yaml 导入，使用新的配置系统
from configs.config import get_rmvl_config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging
from datetime import datetime

# 导入自定义模块
from models.rmvl.seg_unet import SegEnhancedUNet
from common.data.dataset import WatermarkDataset
from common.utils.losses import MultiTaskLoss

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    # 使用新的配置系统
    config = get_rmvl_config()
    
    # 设置日志
    setup_logging(config['logging']['log_dir'])
    logging.info(f"训练配置: {config}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 数据转换
    train_transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 加载数据集
    # 训练集应使用split_ratio参数划分
    train_dataset = WatermarkDataset(
        root_dir=config['data']['train_dir'],
        transform=train_transform,
        mode='train',
        split_ratio=0.8  # ← 确保此参数小于1.0
    )
    
    # 验证集应使用相同的root_dir但mode='val'
    val_dataset = WatermarkDataset(
        root_dir=config['data']['train_dir'],  # ← 注意与训练集相同路径
        transform=val_transform,
        mode='val',
        split_ratio=0.8
    )
    
    logging.info(f"训练样本数: {len(train_dataset)}")
    logging.info(f"验证样本数: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # 初始化模型
    model = SegEnhancedUNet(
        in_channels=3,
        out_channels=3,
        seg_classes=2
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = MultiTaskLoss(
        lambda_seg=config['training']['lambda_seg'],
        lambda_edge=config['training']['lambda_edge']
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config['training']['lr_patience'],
        min_lr=1e-7
    )
    
    # 创建保存目录
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # 训练阶段
        model.train()
        train_losses = {
            'total': 0.0,
            'restoration': 0.0,
            'segmentation': 0.0,
            'edge': 0.0
        }
        
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{config["training"]["epochs"]} [Train]')
        for batch in train_progress:
            wm_images = batch['watermarked'].to(device)
            clean_images = batch['clean'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(wm_images)
            targets = (clean_images, masks)
            losses = criterion(outputs, targets)
            
            # 反向传播
            losses['total'].backward()
            optimizer.step()
            
            # 记录损失
            for key in train_losses:
                train_losses[key] += losses[key].item() * wm_images.size(0)
            
            train_progress.set_postfix({
                'Total Loss': losses['total'].item(),
                'Rest Loss': losses['restoration'].item(),
                'Seg Loss': losses['segmentation'].item(),
                'Edge Loss': losses['edge'].item()
            })
        
        # 计算平均损失
        for key in train_losses:
            train_losses[key] /= len(train_dataset)
        
        logging.info(f'Epoch {epoch}/{config["training"]["epochs"]} - '
                    f'Train Total Loss: {train_losses["total"]:.4f}, '
                    f'Restoration Loss: {train_losses["restoration"]:.4f}, '
                    f'Segmentation Loss: {train_losses["segmentation"]:.4f}, '
                    f'Edge Loss: {train_losses["edge"]:.4f}')
        
        # 验证阶段
        model.eval()
        val_losses = {
            'total': 0.0,
            'restoration': 0.0,
            'segmentation': 0.0,
            'edge': 0.0
        }
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f'Epoch {epoch}/{config["training"]["epochs"]} [Val]')
            for batch in val_progress:
                wm_images = batch['watermarked'].to(device)
                clean_images = batch['clean'].to(device)
                masks = batch['mask'].to(device)
                
                # 前向传播
                outputs = model(wm_images)
                targets = (clean_images, masks)
                losses = criterion(outputs, targets)
                
                # 记录损失
                for key in val_losses:
                    val_losses[key] += losses[key].item() * wm_images.size(0)
                
                val_progress.set_postfix({
                    'Total Loss': losses['total'].item(),
                    'Rest Loss': losses['restoration'].item(),
                    'Seg Loss': losses['segmentation'].item(),
                    'Edge Loss': losses['edge'].item()
                })
        
        # 计算平均损失
        for key in val_losses:
            val_losses[key] /= len(val_dataset)
        
        logging.info(f'Epoch {epoch}/{config["training"]["epochs"]} - '
                    f'Val Total Loss: {val_losses["total"]:.4f}, '
                    f'Restoration Loss: {val_losses["restoration"]:.4f}, '
                    f'Segmentation Loss: {val_losses["segmentation"]:.4f}, '
                    f'Edge Loss: {val_losses["edge"]:.4f}')
        
        # 学习率调度
        scheduler.step(val_losses['total'])
        
        # 保存模型
        model_path = os.path.join(config['training']['save_dir'], f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_path)
        logging.info(f'Model saved to {model_path}')
        
        # 保存最佳模型
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_model_path = os.path.join(config['training']['save_dir'], 'model_best.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'Best model updated: {best_model_path}')

if __name__ == '__main__':
    # 移除 argparse，直接使用配置文件
    main()