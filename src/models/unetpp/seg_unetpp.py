import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 在跳跃连接处添加通道注意力
        def conv_block(in_ch, out_ch, attn=False):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if attn:
                layers.append(ChannelAttention(out_ch))
            return nn.Sequential(*layers)

        # 定义注意力模块
        return self.block(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.upsample(x)

class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()
        self.g_conv = nn.Conv2d(g_channels, inter_channels, kernel_size=1)
        self.x_conv = nn.Conv2d(x_channels, inter_channels, kernel_size=1)
        self.psi_conv = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, g, x):
        g1 = self.g_conv(g)
        x1 = self.x_conv(x)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        psi = self.sigmoid(self.psi_conv(F.relu(g1 + x1)))
        return x * psi

class SegGuidedUnetPP(nn.Module):
    def __init__(self, in_channels=3, img_out_channels=3, seg_out_channels=1):
        super().__init__()
        
        # 编码器
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(512, 1024)
        
        # 图像生成分支
        self.g_up4 = UpSample(1024, 512)
        self.g_att4 = AttentionGate(512, 512, 256)
        self.g_conv4 = ConvBlock(1024, 512)
        
        self.g_up3 = UpSample(512, 256)
        self.g_att3 = AttentionGate(256, 256, 128)
        self.g_conv3 = ConvBlock(512, 256)
        
        self.g_up2 = UpSample(256, 128)
        self.g_att2 = AttentionGate(128, 128, 64)
        self.g_conv2 = ConvBlock(256, 128)
        
        self.g_up1 = UpSample(128, 64)
        self.g_att1 = AttentionGate(64, 64, 32)
        self.g_conv1 = ConvBlock(128, 64)
        
        self.img_out = nn.Conv2d(64, img_out_channels, kernel_size=1)
        
        # 分割分支
        self.s_up3 = UpSample(512, 256)
        self.s_att3 = AttentionGate(256, 128, 128)
        self.s_conv3 = ConvBlock(384, 128)
        
        self.s_up2 = UpSample(128, 64)
        self.s_att2 = AttentionGate(64, 64, 64)
        self.s_conv2 = ConvBlock(128, 64)
        
        self.s_up1 = UpSample(64, 32)
        self.s_conv1 = ConvBlock(32, 32)
        
        self.seg_out = nn.Conv2d(32, seg_out_channels, kernel_size=1)
        self.att_gate1 = AttentionGate(g_channels=64, x_channels=64, inter_channels=32)
        self.att_gate2 = AttentionGate(g_channels=128, x_channels=128, inter_channels=64)
    
    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        
        # 图像生成分支
        g4 = self.g_up4(b)
        e4_att = self.g_att4(g4, e4)
        g4 = torch.cat([g4, e4_att], dim=1)
        g4 = self.g_conv4(g4)
        
        g3 = self.g_up3(g4)
        e3_att = self.g_att3(g3, e3)
        g3 = torch.cat([g3, e3_att], dim=1)
        g3 = self.g_conv3(g3)
        
        g2 = self.g_up2(g3)
        e2_att = self.g_att2(g2, e2)
        g2 = torch.cat([g2, e2_att], dim=1)
        g2 = self.g_conv2(g2)
        
        g1 = self.g_up1(g2)
        e1_att = self.g_att1(g1, e1)
        g1 = torch.cat([g1, e1_att], dim=1)
        g1 = self.g_conv1(g1)
        
        img_out = self.img_out(g1)
        
        # 分割分支
        s3 = self.s_up3(e4)
        e2_att_s = self.s_att3(s3, e2)
        if e2_att_s.shape[-2:] != s3.shape[-2:]:
            e2_att_s = F.interpolate(e2_att_s, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        s3 = torch.cat([s3, e2_att_s], dim=1)
        s3 = self.s_conv3(s3)
        
        s2 = self.s_up2(s3)
        e1_att_s = self.s_att2(s2, e1)
        # 在原有代码基础上增加尺寸校验
        e1_att_s = self.att_gate1(e1, s2)
        
        # 添加自动插值逻辑
        if e1_att_s.shape[-2:] != s2.shape[-2:]:
            e1_att_s = F.interpolate(e1_att_s, size=s2.shape[-2:], mode='bilinear', align_corners=False)
        
        s2 = torch.cat([s2, e1_att_s], dim=1)
        s2 = self.s_conv2(s2)
        
        s1 = self.s_up1(s2)
        s1 = self.s_conv1(s1)
        
        seg_out = torch.sigmoid(self.seg_out(s1))
        
        return img_out, seg_out

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )