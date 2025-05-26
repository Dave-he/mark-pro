import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SegEnhancedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, seg_classes=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seg_classes = seg_classes

        # 编码器
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        
        # 瓶颈层
        self.bottleneck = DoubleConv(512, 1024)
        
        # 解码器（去水印分支）
        self.decoder4 = DoubleConv(1024 + 512, 512)
        self.decoder3 = DoubleConv(512 + 256, 256)
        self.decoder2 = DoubleConv(256 + 128, 128)
        self.decoder1 = DoubleConv(128 + 64, 64)
        
        # 输出层（去水印）
        self.restoration_head = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # 分割分支
        self.seg_decoder4 = DoubleConv(1024 + 512, 512)
        self.seg_decoder3 = DoubleConv(512 + 256, 256)
        self.seg_decoder2 = DoubleConv(256 + 128, 128)
        self.seg_decoder1 = DoubleConv(128 + 64, 64)
        
        # 输出层（分割）
        self.seg_head = nn.Conv2d(64, seg_classes, kernel_size=1)
        
        # 池化和上采样
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # 编码器路径
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # 瓶颈层
        b = self.bottleneck(self.pool(e4))
        
        # 解码器路径（去水印分支）
        d4 = self.decoder4(torch.cat([self.upsample(b), e4], dim=1))
        d3 = self.decoder3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upsample(d2), e1], dim=1))
        
        # 去水印输出
        restoration = self.restoration_head(d1)
        
        # 分割分支
        s4 = self.seg_decoder4(torch.cat([self.upsample(b), e4], dim=1))
        s3 = self.seg_decoder3(torch.cat([self.upsample(s4), e3], dim=1))
        s2 = self.seg_decoder2(torch.cat([self.upsample(s3), e2], dim=1))
        s1 = self.seg_decoder1(torch.cat([self.upsample(s2), e1], dim=1))
        
        # 分割输出
        seg_logits = self.seg_head(s1)
        
        return restoration, seg_logits