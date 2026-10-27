"""Verbatim from notebooks/Final_Version_FDCONV.ipynb, cell 7 (ASPP).
Only additions: the imports that cell 0 provided in the notebook session."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        # 1x1卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 3x3空洞卷积分支，使用不同膨胀率
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rates[0], dilation=dilation_rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rates[1], dilation=dilation_rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rates[2], dilation=dilation_rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 图像级全局特征分支
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 融合层
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)
        # 全局特征处理并上采样
        feat5 = self.global_pool(x)
        feat5 = F.interpolate(feat5, size=size, mode='bilinear', align_corners=True)
        # 拼接所有分支
        x = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        x = self.conv_out(x)
        return x


