'''
Date: 2021-10-16 17:59:30
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-02-28 20:50:56
FilePath: /D2/data/augmentations/augmentation_impl.py
'''
import numpy as np
from PIL import Image
import math
from detectron2.data.transforms import Augmentation, NoOpTransform

from ..transforms.transform import (
    AffineTransform,
    ResizeT,
    AlbumentationsTransform,
    NanoDetT
)

__all__ = [
    "CenterAffine",
    "ResizeAffine",
    "AlbumentationsWrapper",
    "NanoDetAffine"
]

class CenterAffine(Augmentation):
    """
    Affine Transform
    """

    def __init__(self, output_size, rescale=True, size_div=32, pad_val=(114,114,114)):
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
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        if self.rescale:
            output_shape = self.output_size
        else:
            scale = min(self.output_size[0]/img_shape[0],self.output_size[1]/img_shape[1])
            if img_shape[0]>img_shape[1]: # h>w
                new_w = math.ceil(img_shape[1]*scale/self.size_div)*self.size_div
                output_shape = (int(img_shape[0]*scale), new_w)
            else:
                new_h = math.ceil(img_shape[0]*scale/self.size_div)*self.size_div
                output_shape = (new_h, int(img_shape[1]*scale))
        return AffineTransform(img_shape[::-1], output_shape[::-1], self.pad_val)

class ResizeAffine(Augmentation):
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
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        return ResizeT(img_shape[::-1], self.output_size[::-1])

class AlbumentationsWrapper(Augmentation):
    """
    Wrap an augmentor form the albumentations library: https://github.com/albu/albumentations.
    Image, Bounding Box and Segmentation are supported.
    Example:
    .. code-block:: python
        import albumentations as A
        from detectron2.data import transforms as T
        from detectron2.data.transforms.albumentations import AlbumentationsWrapper
        augs = T.AugmentationList([
            AlbumentationsWrapper(A.RandomCrop(width=256, height=256)),
            AlbumentationsWrapper(A.HorizontalFlip(p=1)),
            AlbumentationsWrapper(A.RandomBrightnessContrast(p=1)),
        ])  # type: T.Augmentation
        # Transform XYXY_ABS -> XYXY_REL
        h, w, _ = IMAGE.shape
        bbox = np.array(BBOX_XYXY) / [w, h, w, h]
        # Define the augmentation input ("image" required, others optional):
        input = T.AugInput(IMAGE, boxes=bbox, sem_seg=IMAGE_MASK)
        # Apply the augmentation:
        transform = augs(input)
        image_transformed = input.image  # new image
        sem_seg_transformed = input.sem_seg  # new semantic segmentation
        bbox_transformed = input.boxes   # new bounding boxes
        # Transform XYXY_REL -> XYXY_ABS
        h, w, _ = image_transformed.shape
        bbox_transformed = bbox_transformed * [w, h, w, h]
    """

    def __init__(self, augmentor):
        """
        Args:
            augmentor (albumentations.BasicTransform):
        """
        # super(Albumentations, self).__init__() - using python > 3.7 no need to call rng
        self._aug = augmentor

    def get_transform(self, image):
        do = self._rand_range() < self._aug.p
        if do:
            return AlbumentationsTransform(self._aug)
        else:
            return NoOpTransform()


class NanoDetAffine(Augmentation):
    """
    Affine Transform
    """

    def __init__(self, pipeline, keep_ratio, output_size):
        """
        Args:
            pipeline: pipeline of nanodet
            keep_ratio: resize keep ratio
            output_size(tuple): a tuple represents (width, height) of image
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        """
        generate one `AffineTransform` for input image
        """
        img_shape = img.shape[:2]  # (height, width) 
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        return NanoDetT(img_shape[::-1], self.output_size[::-1], self.pipeline, self.keep_ratio)
