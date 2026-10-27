"""Verbatim from notebooks/Final_Version_FDCONV.ipynb, cell 8.

`ResNetEncoder` (originally `DinoEncoder`) is the paper's encoder: a ResNet-50
backbone with an FFTBlock appended to the end of each stage (class renamed
only; module attribute names and checkpoint keys are unchanged). `Deconv` is
a plain 3x3 convolution helper.
Cross-module imports (resnet50 from cell 1, FFTBlock/FDConv from cell 4) were
provided by the shared notebook session in the original."""
from .fft_block import FFTBlock
from .resnet import resnet50


import torch
import torch.nn as nn
class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Deconv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.conv(x)         # 普通卷积
        return x

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # [batch_size, 64, h/2, w/2]
        self.fdconv0 =  FFTBlock(64, 64)  # [batch_size, 128, h/2, w/2]
       
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)  # [batch_size, 256, h/4, w/4]
        self.fdconv1 = FFTBlock(256, 256)
        
        self.layer2 = backbone.layer2  # [batch_size, 512, h/8, w/8]
        self.fdconv2 = FFTBlock(512, 512)
        
        self.layer3 = backbone.layer3  # [batch_size, 1024, h/16, w/16]
        self.fdconv3 = FFTBlock(1024, 1024)
        
    def forward(self, x):
        x0 = x
        x1 = self.layer0(x0) # [B, 64, h/4, w/4]
        x1 = self.fdconv0(x1)
        
        x2 = self.layer1(x1) # [B, 128, h/8, w/8]
        x2 = self.fdconv1(x2)
       

        x3 = self.layer2(x2) # [B, 320, h/16, w/16]
        x3 = self.fdconv2(x3)
        
        x4 = self.layer3(x3) # [B, 512, h/32, w/32]
        x4 = self.fdconv3(x4)
        
        # 你可以根据需要进一步处理这些特征图，这里仅返回它们
        return x1, x2, x3, x4
