import torch
import numpy as np

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

import torch.nn.functional as F

def ssim(img1, img2, window_size=11, size_average=True):
    # 简化版SSIM计算，实际应用中建议使用专门库
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = F.avg_pool2d(img1, window_size, stride=1)
    mu2 = F.avg_pool2d(img2, window_size, stride=1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = (pred + target - pred * target).sum()
    return (intersection / (union + 1e-8)).item()