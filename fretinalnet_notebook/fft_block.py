"""Verbatim from notebooks/Final_Version_FDCONV.ipynb, cell 4.

This is the paper's Frequency Feature Tuning Block (FFTBlock): convolutional
kernels parameterized as complex spectral bases, input-adaptive modulation
(global/local context), and group-wise magnitude-spectrum refinement.
Called `FDConv` in the original notebook (class renamed only; module
attribute names and checkpoint keys are unchanged)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class FFTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=8):
        super(FFTBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        
        # 稳定化参数初始化
        self.fdw_param = nn.Parameter(
            (torch.randn(groups, out_channels, in_channels, kernel_size, kernel_size, dtype=torch.cfloat) * 0.1)
        )
        
        # KSM调制 (加入激活函数)
        self.ksm_global_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, groups),
            nn.Sigmoid()
        )
        
        self.ksm_local_conv1d = nn.Sequential(
            nn.Conv1d(in_channels, groups, kernel_size=1),
            nn.Sigmoid()
        )
        
        # FBM调制 (加入激活函数)
        self.fbm_conv = nn.Sequential(
            nn.Conv2d(out_channels, groups, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # BatchNorm稳定化
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        B, C, H, W = x.size()
        res=x
        weight_spatial = []
        for g in range(self.groups):
            weight_freq = self.fdw_param[g]
            weight_spatial.append(torch.real(torch.fft.ifft2(weight_freq, s=(self.kernel_size, self.kernel_size))))
        weights = torch.stack(weight_spatial, dim=0)

        global_context = self.ksm_global_fc(x).view(B, self.groups, 1, 1, 1, 1)
        local_context = self.ksm_local_conv1d(x.mean([-2,-1]).unsqueeze(-1)).view(B, self.groups, 1, 1, 1, 1)
        spatial_modulation = global_context * local_context
        weights = weights.unsqueeze(0) * spatial_modulation
        weights = weights.sum(1)

        out = torch.zeros(B, self.out_channels, H, W, device=x.device)
        for b in range(B):
            out[b] = F.conv2d(x[b:b+1], weights[b], padding=self.kernel_size//2)

        freq_bands = torch.fft.fft2(out)
        modulation_maps = self.fbm_conv(out)

        band_size = self.out_channels // self.groups
        modulated_out = torch.zeros_like(freq_bands)
        for g in range(self.groups):
            band_slice = slice(g*band_size, (g+1)*band_size)
            modulation = modulation_maps[:,g:g+1,:,:]
            modulated_out[:, band_slice,:,:] = freq_bands[:,band_slice,:,:]*modulation

        out = torch.real(torch.fft.ifft2(modulated_out))
        out=out+res
        return self.bn(out)