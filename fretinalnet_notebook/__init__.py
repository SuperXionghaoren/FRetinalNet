"""FRetinalNet notebook-exact model, extracted verbatim from notebooks/Final_Version_FDCONV.ipynb.

Class names follow the paper:
  FRetinalNet        = full segmentation model (notebook `SegmentModel`, cell 11)
  FFTBlock           = Frequency Feature Tuning Block (notebook `FDConv`, cell 4)
  FaFBlock           = Frequency Aware Fusion Block (notebook `DualFeatureFusion`, cell 9)
  ResNetEncoder      = ResNet-50 encoder with an FFTBlock per stage (notebook `DinoEncoder`, cell 8)
  CompositeAttention = decoder composite attention (notebook `CombinedAttentionModule`, cell 10)

Module attribute names (and therefore state_dict / checkpoint keys) are unchanged.
"""
from .dtcwt import DTCWT2D
from .encoder import Deconv, ResNetEncoder
from .fft_block import FFTBlock
from .aspp import ASPP
from .faf_block import CBR, FaFBlock
from .attention import ChannelAttention, SpatialAttention, EdgeAttention, CompositeAttention
from .model import FRetinalNet

__all__ = ["FRetinalNet", "ResNetEncoder", "Deconv", "DTCWT2D", "FFTBlock", "ASPP",
           "CBR", "FaFBlock", "CompositeAttention",
           "ChannelAttention", "SpatialAttention", "EdgeAttention"]
