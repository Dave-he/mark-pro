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
        dice = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
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