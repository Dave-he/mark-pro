# ------------------- 模型定义 -------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        # 省略模型结构（同原代码），此处保持一致即可
        self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.enc3 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.enc4 = nn.Conv2d(256, 512, 3, padding=1, stride=2)
        
        self.dec1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = nn.ConvTranspose2d(256+256, 128, 2, stride=2)
        self.dec3 = nn.ConvTranspose2d(128+128, 64, 2, stride=2)
        self.final = nn.Conv2d(64+64, out_channels, 3, padding=1)
        
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        # 省略前向传播（同原代码）
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        e4 = self.relu(self.enc4(e3))
        
        d1 = self.relu(self.dec1(e4))
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.relu(self.dec2(d1))
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.relu(self.dec3(d2))
        d3 = torch.cat([d3, e1], dim=1)
        
        out = self.final(d3)
        return out