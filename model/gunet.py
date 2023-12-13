import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=1)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#Model ##################################################################


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,encoders, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, encoders[0])
        factor = 2 if bilinear else 1
        self.downs = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()
        last=encoders[0] 
        for i in range(1,len(encoders)-1 ):
          self.downs.append(Down(last, encoders[i]))
          last=encoders[i] 
        self.downs.append(Down(last, encoders[i+1] // factor))
        last =  encoders[i+1] 
        for i in range(len(encoders)-1,0,-1):
          self.ups.append(Up(last, encoders[i] // factor, bilinear))
          last=encoders[i] 
        self.ups.append(Up(last, encoders[0], bilinear))
        self.outc = OutConv(encoders[0], n_classes)  

    def forward(self, x):
        x = self.inc(x) 
        res = []
        res.append(x) 
        for i in range(len(self.downs)):
            x=self.downs[i](x)
            res.append(x) 
        for i in range(len(self.ups)):
            nn=res.pop()
            x=self.ups[i](x,nn) 
        logits = self.outc(x)
        return logits 

class CNNCls(nn.Module):
    def __init__(self, n_channels, n_classes,encoders, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, encoders[0])
        factor = 2 if bilinear else 1
        self.downs = torch.nn.ModuleList()
        last=encoders[0]
        for i in range(1,len(encoders)-1 ):
          self.downs.append(Down(last, encoders[i]))
          last=encoders[i]
        self.downs.append(Down(last, encoders[i+1] // factor))
        last =  encoders[i+1]
        self.fc1 = nn.Linear(-1, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.inc(x)
        res = []
        for i in range(len(self.downs)):
            x=self.downs[i](x)
        x = self.fc1(x)
        logits=self.fc2(x)
        return logits

'''
x    = torch.randn(2, 3, 512, 512)
print(x.shape)
unet = UNet(n_channels=3, n_classes=1,encoders=[128,256,512,1024,2048],bilinear=True)
print(unet(x).shape)
y=unet(x)
print(y)
target = torch.randint(0, 2, (10, 572, 572))
pred = torch.tensor(target)

'''



