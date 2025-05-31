import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
# 修复导入路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import get_unet_mask_config
from models.unetpp.unet_plus_plus import UNetPlusPlus
from common.data.dataset_mask import create_datasets
from common.utils.losses import BCEDiceLoss
from common.utils.metrics import dice_coef

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
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
        
        if (i + 1) % config['train']['log_interval'] == 0:
            progress_bar.set_description(
                f"Epoch [{epoch+1}/{config['train']['epochs']}] "
                f"Batch [{i+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Dice: {dice.item():.4f}"
            )
    
    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    
    return avg_loss, avg_dice

def validate(model, val_loader, criterion, device, config):
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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config):
    """训练模型的主函数"""
    best_dice = 0.0
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    
    for epoch in range(config['train']['epochs']):
        # 训练阶段
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config)
        
        # 验证阶段
        val_loss, val_dice = validate(model, val_loader, criterion, device, config)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        
        print(f'Epoch [{epoch+1}/{config["train"]["epochs"]}]')
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
        print('-' * 50)
        
        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), config['train']['model_save_path'])
            print(f'Best model saved with Dice: {best_dice:.4f}')
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(config['train']['output_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, checkpoint_path)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_dices, label='Train Dice')
    plt.plot(val_dices, label='Val Dice')
    plt.title('Dice Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['train']['output_dir'], 'training_curves.png'))
    plt.show()
    
    print(f'Training completed! Best Dice: {best_dice:.4f}')

def train():
    # 使用新的配置系统
    config = get_unet_mask_config()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(config['train']['model_save_path']), exist_ok=True)
    os.makedirs(config['train']['output_dir'], exist_ok=True)
    
    # 初始化模型
    device = torch.device(config['runtime']['device'])
    model = UNetPlusPlus(
        in_channels=config['model']['input_channels'],
        num_classes=config['model']['num_classes'],
        deep_supervision=config['model']['deep_supervision']
    ).to(device)
    
    # 创建数据集
    train_dataset, val_dataset = create_datasets(config)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=4)
    
    # 定义损失函数和优化器
    criterion = BCEDiceLoss()
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['train']['lr']),  # 确保转换为浮点数
        weight_decay=float(config['train']['weight_decay'])  # 确保转换为浮点数
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config)

def main():
    # 使用新的配置系统
    config = get_unet_mask_config()
    
    # 创建必要的目录
    os.makedirs(os.path.dirname(config['train']['model_save_path']), exist_ok=True)
    os.makedirs(config['train']['output_dir'], exist_ok=True)
    
    print("开始训练...")
    train()
    print("训练完成！")

if __name__ == "__main__":
    main()