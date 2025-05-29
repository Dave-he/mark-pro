import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        intersection = (y_pred * y_true).sum()
        dice = (2 * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1 - dice

class MultiTaskLoss(nn.Module):
    def __init__(self, seg_weight=0.3):
        super().__init__()
        self.img_loss = nn.MSELoss()
        self.seg_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.seg_weight = seg_weight
    
    def forward(self, img_pred, seg_pred, img_true, seg_true):
        img_l = self.img_loss(img_pred, img_true)
        seg_l = self.seg_loss(seg_pred, seg_true) + self.dice_loss(seg_pred, seg_true)
        return img_l + self.seg_weight * seg_l


class MultiTaskLossRmvl(nn.Module):
    def __init__(self, lambda_seg=0.5, lambda_edge=0.2):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_edge = lambda_edge
        
        self.restoration_loss = nn.L1Loss()
        self.seg_loss = nn.CrossEntropyLoss()
        
        # 边缘检测算子
        sobel_kernel = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_x', sobel_kernel)
        self.register_buffer('sobel_y', sobel_kernel.transpose(2, 3))

    def compute_edge(self, x):
        """计算图像边缘"""
        if x.dim() == 4 and x.shape[1] == 3:  # RGB图像
            x = x.mean(dim=1, keepdim=True)  # 转为灰度
            
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        
        edge = torch.sqrt(grad_x**2 + grad_y**2)
        return edge

    def forward(self, outputs, targets):
        restoration, seg_logits = outputs
        clean_image, mask = targets
        
        # 计算去水印损失
        rest_loss = self.restoration_loss(restoration, clean_image)
        
        # 计算分割损失
        seg_loss = self.seg_loss(seg_logits, mask.squeeze(1).long())
        
        # 计算边缘损失
        edge_target = self.compute_edge(clean_image)
        edge_pred = self.compute_edge(restoration)
        edge_loss = self.restoration_loss(edge_pred, edge_target)
        
        # 总损失
        total_loss = rest_loss + self.lambda_seg * seg_loss + self.lambda_edge * edge_loss
        
        return {
            'total': total_loss,
            'restoration': rest_loss,
            'segmentation': seg_loss,
            'edge': edge_loss
        }

# 添加边缘感知损失
class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    
    def forward(self, y_pred, y_true):
        # 转换为灰度
        if y_pred.shape[1] > 1:
            y_pred = 0.299 * y_pred[:, 0:1] + 0.587 * y_pred[:, 1:2] + 0.114 * y_pred[:, 2:3]
        if y_true.shape[1] > 1:
            y_true = 0.299 * y_true[:, 0:1] + 0.587 * y_true[:, 1:2] + 0.114 * y_true[:, 2:3]
        
        # 计算边缘
        pred_edge_x = F.conv2d(y_pred, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(y_pred, self.sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)
        
        true_edge_x = F.conv2d(y_true, self.sobel_x, padding=1)
        true_edge_y = F.conv2d(y_true, self.sobel_y, padding=1)
        true_edge = torch.sqrt(true_edge_x ** 2 + true_edge_y ** 2 + 1e-6)
        
        return F.l1_loss(pred_edge, true_edge)

# 组合损失函数
class CombinedLoss(nn.Module):
    def __init__(self, seg_weight=0.3, edge_weight=0.1):
        super().__init__()
        self.img_loss = nn.L1Loss()
        self.seg_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.edge_loss = EdgeLoss()
        self.seg_weight = seg_weight
        self.edge_weight = edge_weight
    
    def forward(self, img_pred, seg_pred, img_true, seg_true):
        img_l = self.img_loss(img_pred, img_true)
        seg_l = self.seg_loss(seg_pred, seg_true) + self.dice_loss(seg_pred, seg_true)
        edge_l = self.edge_loss(img_pred, img_true)
        return img_l + self.seg_weight * seg_l + self.edge_weight * edge_l

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
    
    def forward(self, logits, targets):
        # 处理维度不匹配：如果 logits 有额外的通道维度，则压缩它
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)  # 从 [B, 1, H, W] 变为 [B, H, W]
        
        # 确保 targets 也是正确的维度
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)
            
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice