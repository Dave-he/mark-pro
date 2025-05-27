import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from yacs.config import CfgNode as CN
from models.unetpp.unet_plus_plus import UNetPlusPlus
from common.data.dataset_mask import create_datasets
from common.utils.losses import BCEDiceLoss
from common.utils.metrics import dice_coef
from configs.unetmask import cfg


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, masks) in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        # 计算dice系数
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            dice = dice_coef(preds, masks)
        
        total_loss += loss.item()
        total_dice += dice.item()
        
        if (i + 1) % cfg.TRAIN.LOG_INTERVAL == 0:
            progress_bar.set_description(
                f"Epoch [{epoch+1}/{cfg.TRAIN.EPOCHS}] "
                f"Batch [{i+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Dice: {dice.item():.4f}"
            )
    
    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    
    return avg_loss, avg_dice

def validate(model, val_loader, criterion, device, cfg):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, (images, masks) in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 计算dice系数
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            dice = dice_coef(preds, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            
            progress_bar.set_description(
                f"Val Batch [{i+1}/{len(val_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Dice: {dice.item():.4f}"
            )
    
    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    
    return avg_loss, avg_dice

def train():
    # 创建输出目录
    os.makedirs(os.path.dirname(cfg.TRAIN.MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(cfg.TRAIN.OUTPUT_DIR, exist_ok=True)
    
    # 初始化模型
    device = torch.device(cfg.DEVICE)
    model = UNetPlusPlus(
        in_channels=cfg.MODEL.INPUT_CHANNELS,
        num_classes=cfg.MODEL.NUM_CLASSES,
        deep_supervision=cfg.MODEL.DEEP_SUPERVISION
    ).to(device)
    
    # 创建数据集
    train_dataset, val_dataset = create_datasets(cfg)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 定义损失函数和优化器
    criterion = BCEDiceLoss(
        bce_weight=cfg.LOSS.BCE_WEIGHT,
        smooth=cfg.LOSS.DICE_SMOOTH
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.TRAIN.LR,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    
    print(f"开始训练 {cfg.MODEL.NAME} 模型...")
    print(f"训练集: {len(train_dataset)} 张图像")
    print(f"验证集: {len(val_dataset)} 张图像")
    
    for epoch in range(cfg.TRAIN.EPOCHS):
        # 训练一个epoch
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg)
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        
        # 验证
        val_loss, val_dice = validate(model, val_loader, criterion, device, cfg)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 打印训练信息
        print(f"Epoch [{epoch+1}/{cfg.TRAIN.EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.TRAIN.MODEL_SAVE_PATH)
            print(f"保存最佳模型: {cfg.TRAIN.MODEL_SAVE_PATH} (Val Loss: {val_loss:.4f})")
        
        # 定期保存模型
        if (epoch + 1) % cfg.TRAIN.SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), os.path.join(
                os.path.dirname(cfg.TRAIN.MODEL_SAVE_PATH),
                f"unet_plus_plus_epoch_{epoch+1}.pth"
            ))
    
    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss History')
    
    plt.subplot(122)
    plt.plot(train_dices, label='Train Dice')
    plt.plot(val_dices, label='Val Dice')
    plt.legend()
    plt.title('Dice Coefficient History')
    
    plt.savefig(os.path.join(cfg.TRAIN.OUTPUT_DIR, 'training_history.png'))
    plt.close()
    
    print(f"训练完成！最佳验证损失: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()