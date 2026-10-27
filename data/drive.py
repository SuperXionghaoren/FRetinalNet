"""DRIVE dataset classes and preprocessing transforms (512x512 resize,
CLAHE, threshold binarization at 0.5)."""
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


#############################################
# 数据集定义
#############################################
class DRIVE_Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = [os.path.join(images_dir, x) for x in sorted(os.listdir(images_dir)) if x.endswith('.tif')]
        self.masks = [os.path.join(masks_dir, x.replace('training.tif', 'manual1.gif'))
                      for x in sorted(os.listdir(images_dir)) if x.endswith('.tif')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 转换为灰度图

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
# 对 mask 做额外的阈值二值化处理，阈值设置为 0.5
        mask = (mask > 0.5).float()
        return image, mask
    
class DRIVE_Test_Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = [os.path.join(images_dir, x) for x in sorted(os.listdir(images_dir)) if x.endswith('.tif')]
        self.masks = [os.path.join(masks_dir, x.replace('test.tif', 'manual1.gif'))
                      for x in sorted(os.listdir(images_dir)) if x.endswith('.tif')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 转换为灰度图

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
# 对 mask 做额外的阈值二值化处理，阈值设置为 0.5
        mask = (mask > 0.5).float()
        return image, mask

#############################################
# 自定义 CLAHE 增强 transform
#############################################
class CLAHE_Transform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def __call__(self, img):
        # 如果输入是 PIL 图像，转换为 numpy 数组
        if isinstance(img, Image.Image):
            img = np.array(img)
        # 对 RGB 图像进行 CLAHE 增强
        if len(img.shape) == 3:
            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(img_lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            l = clahe.apply(l)
            img_lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img)

#############################################
# transform 定义
#############################################
transform = transforms.Compose([
    transforms.Resize((512, 512)),                        # 调整图像大小为 512x512
    CLAHE_Transform(clip_limit=2.0, tile_grid_size=(8, 8)), # CLAHE 增强
    # 如有需要，可加入 FrangiTransform（示例代码中注释掉）
    transforms.ToTensor(),                                # 转换为 tensor
])
