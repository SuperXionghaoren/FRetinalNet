"""Verbatim from notebooks/Final_Version_FDCONV.ipynb, cell 2 (DTCWT2D block)."""
import torch
import torch.nn as nn


#############################################
# DTCWT2D 模型定义
#############################################
class DTCWT2D(nn.Module):
    def __init__(self, in_channels, out_channels, trainable=False, fuse_mode='magnitude', fuse_low_high=False):
        """
        简化版 DTCWT2D 模型，采用组卷积和 Haar 小波滤波器实现低频与高频分解。
        """
        super(DTCWT2D, self).__init__()
        self.fuse_mode = fuse_mode
        self.fuse_low_high = fuse_low_high
        
        # 使用组卷积，每个通道独立滤波，kernel_size=2, stride=2 使得尺寸下采样一半
        self.low_conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False, groups=in_channels)
        self.high_real_conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False, groups=in_channels)
        self.high_imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False, groups=in_channels)
        
        # Haar 小波滤波器
        low_filter = torch.tensor([[0.5, 0.5],
                                   [0.5, 0.5]], dtype=torch.float32)
        high_real_filter = torch.tensor([[0.5, 0.5],
                                         [-0.5, -0.5]], dtype=torch.float32)
        high_imag_filter = torch.tensor([[0.5, -0.5],
                                         [0.5, -0.5]], dtype=torch.float32)
        
        # 扩展滤波器尺寸至 (in_channels, 1, 2, 2)
        low_kernel = low_filter.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        high_real_kernel = high_real_filter.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        high_imag_kernel = high_imag_filter.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        
        # 注册缓冲区，使得模型转移设备时滤波器也随之移动
        self.register_buffer("low_kernel", low_kernel)
        self.register_buffer("high_real_kernel", high_real_kernel)
        self.register_buffer("high_imag_kernel", high_imag_kernel)
        
        with torch.no_grad():
            self.low_conv.weight.copy_(self.low_kernel)
            self.high_real_conv.weight.copy_(self.high_real_kernel)
            self.high_imag_conv.weight.copy_(self.high_imag_kernel)
            
        # 冻结参数
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        low = self.low_conv(x)
        high_real = self.high_real_conv(x)
        high_imag = self.high_imag_conv(x)
        # 通过幅值融合得到高频特征
        fused_high = torch.sqrt(high_real ** 2 + high_imag ** 2)
        return low, fused_high
