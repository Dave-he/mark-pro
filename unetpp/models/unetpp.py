import torch
import torch.nn as nn

# ====================== Unet++ 模型定义 ======================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.up(x)


class UnetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UnetPlusPlus, self).__init__()
        
        # 编码器
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # 瓶颈
        self.bottleneck = ConvBlock(512, 1024)
        
        # 解码器（带跳跃连接）
        self.up4 = UpSample(1024, 512)
        self.conv4 = ConvBlock(1024, 512)  # 拼接后通道数
        
        self.up3 = UpSample(512, 256)
        self.conv3 = ConvBlock(512, 256)
        
        self.up2 = UpSample(256, 128)
        self.conv2 = ConvBlock(256, 128)
        
        self.up1 = UpSample(128, 64)
        self.conv1 = ConvBlock(128, 64)
        
        # 输出层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        print(f"输入x shape: {x.shape}")
        enc1 = self.enc1(x)
        print(f"enc1: {enc1.shape}")
        enc2 = self.enc2(self.pool1(enc1))
        print(f"enc2: {enc2.shape}")
        enc3 = self.enc3(self.pool2(enc2))
        print(f"enc3: {enc3.shape}")
        enc4 = self.enc4(self.pool3(enc3))
        print(f"enc4: {enc4.shape}")
        bottleneck = self.bottleneck(self.pool4(enc4))
        print(f"bottleneck: {bottleneck.shape}")
        dec4 = self.up4(bottleneck)
        print(f"dec4 up: {dec4.shape}")
        dec4 = torch.cat([dec4, enc4], dim=1)
        print(f"dec4 cat: {dec4.shape}")
        dec4 = self.conv4(dec4)
        dec3 = self.up3(dec4)
        print(f"dec3 up: {dec3.shape}")
        dec3 = torch.cat([dec3, enc3], dim=1)
        print(f"dec3 cat: {dec3.shape}")
        dec3 = self.conv3(dec3)
        dec2 = self.up2(dec3)
        print(f"dec2 up: {dec2.shape}")
        dec2 = torch.cat([dec2, enc2], dim=1)
        print(f"dec2 cat: {dec2.shape}")
        dec2 = self.conv2(dec2)
        dec1 = self.up1(dec2)
        print(f"dec1 up: {dec1.shape}")
        dec1 = torch.cat([dec1, enc1], dim=1)
        print(f"dec1 cat: {dec1.shape}")
        dec1 = self.conv1(dec1)
        output = self.final_conv(dec1)
        print(f"输出output: {output.shape}")
        return output

