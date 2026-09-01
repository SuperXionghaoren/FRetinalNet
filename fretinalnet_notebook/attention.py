"""Verbatim from notebooks/Final_Version_FDCONV.ipynb, cell 10.

`CompositeAttention` (originally `CombinedAttentionModule`) is the paper's
decoder "composite attention": channel attention -> spatial attention ->
edge attention with a residual connection (class renamed only; module
attribute names and checkpoint keys are unchanged)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x  # [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv1(x_cat)
        return x0 * self.sigmoid(attention)


# 边缘注意力模块
class EdgeAttention(nn.Module):
    def __init__(self, in_channels):
        super(EdgeAttention, self).__init__()
        self.conv_edge = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        edge_attention = self.conv_edge(x)
        enhanced_feature = x * edge_attention
        return enhanced_feature


# 组合注意力模块：先通道注意力+空间注意力，再边缘注意力
class CompositeAttention(nn.Module):
    def __init__(self, in_channels, ratio=16, spatial_kernel_size=7):
        super(CompositeAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)
        self.edge_attention = EdgeAttention(in_channels)

    def forward(self, x):
        # 先通过通道注意力
        res=x
        x = self.channel_attention(x)
        # 接着通过空间注意力
        x = self.spatial_attention(x)
        # 最后利用边缘注意力增强特征
        x = self.edge_attention(x)
        x=x+res
        return x
