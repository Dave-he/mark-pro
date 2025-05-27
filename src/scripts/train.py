
# 添加混合精度训练
from torch.cuda.amp import autocast, GradScaler


# 使用PyTorch Lightning或自定义Trainer类优化训练流程
class WatermarkRemovalTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_val_loss = float('inf')
        self.writer = SummaryWriter(log_dir='runs/watermark_remover')
        
 def train_epoch(self, train_loader):
    self.model.train()
    metrics = {'loss': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'iou': 0.0}
    scaler = GradScaler()
    
    for inputs, targets, masks in train_loader:
        inputs, targets, masks = inputs.to(self.device), targets.to(self.device), masks.to(self.device)
        
        self.optimizer.zero_grad()
        
        # 使用混合精度
        with autocast():
            img_out, seg_out = self.model(inputs)
            loss = self.criterion(img_out, seg_out, targets, masks)
        
        # 使用scaler进行反向传播
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()

        # 使用tqdm显示进度条
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Training")
        
        for inputs, targets, masks in pbar:
            # 训练步骤
            # ...
            
            # 更新进度条
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        return metrics
    
    def validate(self, val_loader):
        # 验证逻辑
        # ...
        
    def fit(self, train_loader, val_loader, epochs):
        # 训练循环
        # ...