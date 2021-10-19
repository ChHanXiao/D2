'''
Date: 2021-10-16 17:58:48
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2021-10-17 11:17:18
FilePath: /D2/data/transforms/transform.py
'''
from typing import Any, Dict, Union, Tuple

import cv2
import numpy as np

from detectron2.data.transforms import (
    Transform,
    HFlipTransform,
    VFlipTransform,
    ResizeTransform
)

__all__ = [
    "AffineTransform",
]

@Transform.register_type("warp_matrix")
def apply_warp_matrix(transform: Transform, warp_matrix: np.ndarray) -> np.ndarray:
    """
    Add a new field to save warp matrix .
    """

    return warp_matrix

class AffineTransform(Transform):
    """
    Augmentation Affine
    """

    def __init__(self, src, dst, output_size):
        """
        output_size:(w, h)
        """
        super().__init__()
        affine = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply AffineTransform for the image(s).
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the image(s) after applying affine transform.
        """
        return cv2.warpAffine(img, self.affine, self.output_size, flags=cv2.INTER_LINEAR)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Affine the coordinates.
        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.
        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H 1 - y)`.
        """
        # aug_coord (N, 3) shape, self.affine (2, 3) shape
        w, h = self.output_size
        aug_coords = np.column_stack((coords, np.ones(coords.shape[0])))
        coords = np.dot(aug_coords, self.affine.T)
        coords[..., 0] = np.clip(coords[..., 0], 0, w - 1)
        coords[..., 1] = np.clip(coords[..., 1], 0, h - 1)
        return coords

    def apply_warp_matrix(self, warp_matrix: np.ndarray) -> np.ndarray:
        C = np.eye(3)
        C[:2,:] = self.affine
        warp_matrix = C @ warp_matrix

        return warp_matrix
