"""Verbatim from notebooks/Final_Version_FDCONV.ipynb, cell 11 (SegmentModel),
renamed to FRetinalNet per the paper.

Only changes: the implicit torch.load("STAGE_FDCONV.pth") inside __init__ is
removed (use load_stage1() explicitly); constructor call sites use the
paper block names (FFTBlock / FaFBlock / CompositeAttention / ResNetEncoder).
Module ATTRIBUTE names (self.dino_encoder, self.triplefusion*, self.fdconv*,
self.comb_attention*, ...) are untouched so existing checkpoints load as-is.

Paper terminology note: each decoder stage (up_x? + fuse? + comb_attention?)
corresponds to a DeBlock in the paper (up-sampling module + feature fusion
unit + composite attention); the _l variants form the auxiliary
frequency-domain decoder."""
import torch
import torch.nn as nn

from .aspp import ASPP
from .attention import CompositeAttention
from .dtcwt import DTCWT2D
from .encoder import Deconv, ResNetEncoder
from .faf_block import FaFBlock
from .fft_block import FFTBlock




# 完整的 SegmentModel 定义
class FRetinalNet(nn.Module):
    def __init__(self, num_classes=1):
        super(FRetinalNet, self).__init__()
        self.dtcwt = DTCWT2D(in_channels=3, out_channels=3, trainable=False)
        self.highdtcwt=DTCWT2D(in_channels=3, out_channels=3, trainable=False)
        self.lowdtcwt=DTCWT2D(in_channels=3, out_channels=3, trainable=False)
        self.proj=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_l = Deconv(in_channels=9, out_channels=3)
        self.conv_h = Deconv(in_channels=6, out_channels=3)
        self.dino_encoder = ResNetEncoder()
        self.dino_encoder_l= ResNetEncoder()
        self.triplefusion4 = FaFBlock(512*2, 512*2, 2)
        self.triplefusion3 = FaFBlock(512, 512, 2)
        self.triplefusion2 = FaFBlock(256, 256, 2)
        self.triplefusion1 = FaFBlock(64, 64, 2)

        self.aspp= ASPP(1024,1024)
        self.aspp_l= ASPP(1024,1024)

        
        self.triplefusion4_l = FaFBlock(512*2, 512*2, 2)
        self.triplefusion3_l = FaFBlock(512, 512, 2)
        self.triplefusion2_l = FaFBlock(256, 256, 2)
        self.triplefusion1_l = FaFBlock(64, 64, 2)

        self.fdconv4 = FFTBlock(512, 512)
        self.fdconv3 = FFTBlock(256, 256)
        self.fdconv2 = FFTBlock(64, 64)
        
        self.fdconv4_l = FFTBlock(512, 512)
        self.fdconv3_l = FFTBlock(256, 256)
        self.fdconv2_l = FFTBlock(64, 64)

        self.up_x4 = nn.Sequential(
            nn.Conv2d(1024, 256*2*4, kernel_size=3, padding=1),  # 输出 2048 通道
            nn.PixelShuffle(2),  # 上采样因子 2，将通道数降为 2048/4 = 512
            nn.BatchNorm2d(256*2),  # 即 BatchNorm2d(512)
            nn.ReLU(inplace=True)
        )

        # fuse3 层保持不变（如果不需要上采样，仅做特征融合）
        self.fuse3 = nn.Sequential(
            nn.Conv2d(1024, 256*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(256*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256*2, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.comb_attention3=CompositeAttention(in_channels=512, ratio=4, spatial_kernel_size=7)
        # Stage 2: 上采样到 x2 (32x32)
        # 原来注释里的 Deconv(in_channels=512, out_channels=128*2*4) 表示目标为 128*2=256 通道，故卷积输出 256*4 = 1024 通道
        self.up_fuse3 = nn.Sequential(
            nn.Conv2d(512, 128*2*4, kernel_size=3, padding=1),  # 输出 1024 通道
            nn.PixelShuffle(2),  # 上采样后输出 1024/4 = 256 通道
            nn.BatchNorm2d(128*2),  # BatchNorm2d(256)
            nn.ReLU(inplace=True)
        )

        self.fuse2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.comb_attention2=CompositeAttention(in_channels=256, ratio=4, spatial_kernel_size=7)
        # Stage 3: 上采样到 x1 (64x64)
        # 原先 Deconv(in_channels=128*2, out_channels=64*4) 表示目标为 64 通道，所以卷积输出 64*4 = 256 通道
        self.up_fuse2 = nn.Sequential(
            nn.Conv2d(128*2, 64*4, kernel_size=3, padding=1),  # 输入 128*2 = 256 通道, 输出 256 通道
            nn.PixelShuffle(2),  # 上采样后输出 256/4 = 64 通道
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.comb_attention1=CompositeAttention(in_channels=64, ratio=4, spatial_kernel_size=7)
        # Stage 4: 上采样到原始输入尺寸 (256x256)
        # 原注释里的 Deconv(in_channels=64, out_channels=num_classes*4) 表示目标为 num_classes 通道，所以卷积输出 num_classes*4 通道
        self.up_final = nn.Sequential(
            nn.Conv2d(64, num_classes*4, kernel_size=3, padding=1),  # 输出 num_classes*4 通道
            nn.PixelShuffle(2),  # 上采样后输出 num_classes 通道
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, kernel_size=1)  # 进一步映射到类别数（可选，根据需要调整）
        )
        self.comb_attention_final=CompositeAttention(in_channels=num_classes, ratio=1, spatial_kernel_size=7)
        self.conv_fusion = Deconv(in_channels=1*num_classes, out_channels=num_classes)
        
        self.up_x4_l = nn.Sequential(
            nn.Conv2d(1024, 256*2*4, kernel_size=3, padding=1),  # 输出 2048 通道
            nn.PixelShuffle(2),  # 上采样因子 2，将通道数降为 2048/4 = 512
            nn.BatchNorm2d(256*2),  # 即 BatchNorm2d(512)
            nn.ReLU(inplace=True)
        )

        # fuse3 层保持不变（如果不需要上采样，仅做特征融合）
        self.fuse3_l = nn.Sequential(
            nn.Conv2d(1024, 256*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(256*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256*2, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.comb_attention3_l=CompositeAttention(in_channels=512, ratio=4, spatial_kernel_size=7)
        # Stage 2: 上采样到 x2 (32x32)
        # 原来注释里的 Deconv(in_channels=512, out_channels=128*2*4) 表示目标为 128*2=256 通道，故卷积输出 256*4 = 1024 通道
        self.up_fuse3_l = nn.Sequential(
            nn.Conv2d(512, 128*2*4, kernel_size=3, padding=1),  # 输出 1024 通道
            nn.PixelShuffle(2),  # 上采样后输出 1024/4 = 256 通道
            nn.BatchNorm2d(128*2),  # BatchNorm2d(256)
            nn.ReLU(inplace=True)
        )

        self.fuse2_l=nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.comb_attention2_l=CompositeAttention(in_channels=256, ratio=4, spatial_kernel_size=7)
        # Stage 3: 上采样到 x1 (64x64)
        # 原先 Deconv(in_channels=128*2, out_channels=64*4) 表示目标为 64 通道，所以卷积输出 64*4 = 256 通道
        self.up_fuse2_l = nn.Sequential(
            nn.Conv2d(128*2, 64*4, kernel_size=3, padding=1),  # 输入 128*2 = 256 通道, 输出 256 通道
            nn.PixelShuffle(2),  # 上采样后输出 256/4 = 64 通道
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.fuse1_l = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.comb_attention1_l=CompositeAttention(in_channels=64, ratio=4, spatial_kernel_size=7)
        # Stage 4: 上采样到原始输入尺寸 (256x256)
        # 原注释里的 Deconv(in_channels=64, out_channels=num_classes*4) 表示目标为 num_classes 通道，所以卷积输出 num_classes*4 通道
        self.up_final_l = nn.Sequential(
            nn.Conv2d(64, num_classes*4, kernel_size=3, padding=1),  # 输出 num_classes*4 通道
            nn.PixelShuffle(2),  # 上采样后输出 num_classes 通道
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, kernel_size=1)  # 进一步映射到类别数（可选，根据需要调整）
        )
        self.comb_attention_final_l=CompositeAttention(in_channels=num_classes, ratio=1, spatial_kernel_size=7)
        self.conv_fusion_l = Deconv(in_channels=1*num_classes, out_channels=num_classes)
        
    def forward(self, x):
        # 先通过 DTCWT2D 层预处理输入
        low, high = self.dtcwt(x)
        # DTCWT2D 会使输入下采样一半，若 x 尺寸为 256x256，则 low 尺寸为 128x128
        # 使用上采样将 low 恢复到 256x256
        low=self.proj(low)
        high=self.proj(high)
        low_low,low_high=self.lowdtcwt(low)
        high_low,high_high=self.highdtcwt(high)
        low_low=self.proj(low_low)
        low_high=self.proj(low_high)
        high_low=self.proj(high_low)
        high_high=self.proj(high_high) 
        x_low = torch.cat([x, low,high_low], dim=1)  # 通道维度拼接    
        x_low=self.conv_l(x_low)

        x1,x2,x3,x4=self.dino_encoder(x)
        x1_low,x2_low,x3_low,x4_low=self.dino_encoder_l(x_low)
        
        x4=self.aspp(x4)
        x4_low=self.aspp_l(x4_low)
        x4=self.triplefusion4(x4,x4_low)
        x3=self.triplefusion3(x3,x3_low)
        x2=self.triplefusion2(x2,x2_low)
        x1=self.triplefusion1(x1,x1_low)

        x4_low=self.triplefusion4_l(x4_low,x4)
        x3_low=self.triplefusion3_l(x3_low,x3)
        x2_low=self.triplefusion2_l(x2_low,x2)
        x1_low=self.triplefusion1_l(x1_low,x1)
        
    
        #原始分支
        up_x4=self.up_x4(x4)
        x3=self.fdconv4(x3)
        fuse3=torch.cat([up_x4,x3],dim=1)
        fuse3=self.fuse3(fuse3)
        fuse3=self.comb_attention3(fuse3)
        up_fuse3=self.up_fuse3(fuse3)
        x2=self.fdconv3(x2)
        fuse2=torch.cat([up_fuse3,x2],dim=1)
        fuse2=self.fuse2(fuse2)
        fuse2=self.comb_attention2(fuse2)
        up_fuse2=self.up_fuse2(fuse2)
        x1=self.fdconv2(x1)
        fuse1=torch.cat([up_fuse2,x1],dim=1)
        fuse1=self.fuse1(fuse1)
        fuse1=self.comb_attention1(fuse1)
        up_final=self.up_final(fuse1)
        up_final=self.comb_attention_final(up_final)
        fused_out = self.conv_fusion(up_final)
        

        #x_low分支
        up_x4_low=self.up_x4_l(x4_low)
        x3_low=self.fdconv4_l(x3_low)
        fuse3_low=torch.cat([up_x4_low,x3_low],dim=1)
        fuse3_low=self.fuse3_l(fuse3_low)
        fuse3_low=self.comb_attention3_l(fuse3_low)
        up_fuse3_low=self.up_fuse3_l(fuse3_low)
        x2_low=self.fdconv3_l(x2_low)
        fuse2_low=torch.cat([up_fuse3_low,x2_low],dim=1)
        fuse2_low=self.fuse2_l(fuse2_low)
        fuse2_low=self.comb_attention2_l(fuse2_low)
        up_fuse2_low=self.up_fuse2_l(fuse2_low)
        x1_low=self.fdconv2_l(x1_low)
        fuse1_low=torch.cat([up_fuse2_low,x1_low],dim=1)
        fuse1_low=self.fuse1_l(fuse1_low)
        fuse1_low=self.comb_attention1_l(fuse1_low)
        up_final_low=self.up_final_l(fuse1_low)
        up_final_low=self.comb_attention_final_l(up_final_low)
        fused_out_low = self.conv_fusion_l(up_final_low)

        return fused_out,fused_out_low

    @torch.no_grad()
    def load_stage1(self, ckpt_path, strict=False):
        """Notebook behavior: stage-1 encoder weights loaded into both encoders (strict=False)."""
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.dino_encoder.load_state_dict(state_dict, strict=strict)
        self.dino_encoder_l.load_state_dict(state_dict, strict=strict)
        return self


# # 测试模型
# if __name__ == "__main__":
#     #device
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = SegmentModel(num_classes=1).to(device)
#     # 构造一个输入张量，尺寸 [B, 3, 256, 256]
#     x = torch.randn(2, 3, 512, 512).to(device)
#     ll_out,fused_out = model(x)
#     print("输出尺寸：", fused_out.shape)  # 预期输出：[1, 1, 256, 256]
