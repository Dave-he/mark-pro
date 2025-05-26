import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
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