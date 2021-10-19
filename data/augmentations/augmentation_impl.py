'''
Date: 2021-10-16 17:59:30
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2021-10-16 18:15:52
FilePath: /D2/data/augmentations/augmentation_impl.py
'''
import numpy as np
from PIL import Image

from detectron2.data.transforms import Augmentation, NoOpTransform

from ..transforms.transform import (
    AffineTransform,
)

__all__ = [
    "CenterAffine",
]


class CenterAffine(Augmentation):
    """
    Affine Transform
    """

    def __init__(self, output_size):
        """
        Args:
            output_size(tuple): a tuple represents (width, height) of image
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        """
        generate one `AffineTransform` for input image
        """
        img_shape = img.shape[:2]  # (height, width) 
        center, scale = self.generate_center_and_scale(img_shape)
        src, dst = self.generate_src_and_dst(center, scale, self.output_size)
        return AffineTransform(src, dst, self.output_size)

    def generate_center_and_scale(self, img_shape):
        r"""
        generate center and scale for image randomly
        Args:
            shape(tuple): a tuple represents (height, width) of image
        """
        height, width = img_shape
        center = np.array([width / 2, height / 2], dtype=np.float32)
        scale = float(max(img_shape))
        return center, scale

    @staticmethod
    def generate_src_and_dst(center, size, output_size):
        r"""
        generate source and destination for affine transform
        """
        if not isinstance(size, np.ndarray) and not isinstance(size, list):
            size = np.array([size, size], dtype=np.float32)
        src = np.zeros((3, 2), dtype=np.float32)
        src_w = size[0]
        src_dir = [0, src_w * -0.5]
        src[0, :] = center
        src[1, :] = src[0, :] + src_dir
        src[2, :] = src[1, :] + (src_dir[1], -src_dir[0])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst_w, dst_h = output_size
        dst_dir = [0, dst_w * -0.5]
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = dst[0, :] + dst_dir
        dst[2, :] = dst[1, :] + (dst_dir[1], -dst_dir[0])

        return src, dst
