import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from common.data.dataset import create_dataloaders
from models.unetpp.seg_unetpp import SegGuidedUnetPP  # 添加相对路径导入
from common.utils.losses import MultiTaskLoss
from common.utils.metrics import psnr, ssim, iou
from configs.default import cfg
import logging

def train():
    # 创建保存目录
    os.makedirs(cfg.model.save_dir, exist_ok=True)
    
    # 设备配置改为从yaml读取
    device = torch.device(cfg.train.device)


    logging.info(f"使用设备: {device}")
    
    # 数据加载器
    train_loader, val_loader = create_dataloaders()
    
    # 模型
    model = SegGuidedUnetPP().to(device)
    
    # 损失函数和优化器
    criterion = MultiTaskLoss(seg_weight=cfg.train.seg_loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), 
        lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4,
        total_steps=cfg.train.epochs * len(train_loader),
        pct_start=0.3)
    
    # trainer = TorchTrainer(
    #     model=model,
    #     criterion=CombinedLoss(),  # 组合损失函数
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     metrics=[SSIM(), PSNR()],  # 新增评估指标
    #     amp=True,  # 启用混合精度
    #     gradient_clip=0.5
    # )
    
    # TensorBoard 日志
    writer = SummaryWriter(log_dir=cfg.train.log_dir)
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(cfg.train.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_psnr = 0.0
        train_ssim = 0.0
        train_iou = 0.0
        
        for inputs, targets, masks in train_loader:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            
            optimizer.zero_grad()
            img_out, seg_out = model(inputs)
            
            loss = criterion(img_out, seg_out, targets, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_psnr += psnr(img_out, targets).item() * inputs.size(0)
            train_ssim += ssim(img_out, targets).item() * inputs.size(0)
            train_iou += iou(seg_out, masks) * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_psnr /= len(train_loader.dataset)
        train_ssim /= len(train_loader.dataset)
        train_iou /= len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for inputs, targets, masks in val_loader:
                inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
                
                img_out, seg_out = model(inputs)
                loss = criterion(img_out, seg_out, targets, masks)
                
                val_loss += loss.item() * inputs.size(0)
                val_psnr += psnr(img_out, targets).item() * inputs.size(0)
                val_ssim += ssim(img_out, targets).item() * inputs.size(0)
                val_iou += iou(seg_out, masks) * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_psnr /= len(val_loader.dataset)
        val_ssim /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 生成模型名（示例：unetpp_img512x512_sw0.3.pth）
        imgsize = f"{cfg.data.image_size[0]}x{cfg.data.image_size[1]}"
        segw = cfg.train.seg_loss_weight
        model_name = f"unetpp_img{imgsize}_sw{segw}.pth"
        best_model_path = os.path.join(cfg.model.save_dir, model_name)
        # 保存模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
        
        if (epoch + 1) % cfg.train.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(cfg.model.save_dir, f'epoch_{epoch+1}.pth'))
        
        # 打印日志
        print(f'Epoch {epoch+1}/{cfg.train.epochs}')
        print(f'Train Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}, IoU: {train_iou:.4f}')
        print(f'Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}, IoU: {val_iou:.4f}')
        print('-' * 50)
        
        # 写入TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('PSNR/train', train_psnr, epoch)
        writer.add_scalar('PSNR/val', val_psnr, epoch)
        writer.add_scalar('SSIM/train', train_ssim, epoch)
        writer.add_scalar('SSIM/val', val_ssim, epoch)
        writer.add_scalar('IoU/train', train_iou, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)
    
    writer.close()
    print(f'Training completed. Best validation loss: {best_val_loss:.4f}')

if __name__ == "__main__":
    train()