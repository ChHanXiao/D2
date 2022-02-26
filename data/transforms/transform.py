'''
Date: 2021-10-16 17:58:48
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-02-15 22:27:08
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
    "ResizeT",
    "AlbumentationsTransform"
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

    def __init__(self, src, dst, borderValue):
        """
        src:(w, h) raw_img size
        dst:(w,h) out img size
        """
        super().__init__()

        def letterbox_warp(img_size, expected_size):
            iw,ih = img_size
            ew,eh = expected_size
            scale = min(eh / ih, ew / iw)
            nh = int(ih * scale)
            nw = int(iw * scale)
            smat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float32)
            top = (eh - nh) // 2
            bottom = eh - nh - top
            left = (ew - nw) // 2
            right = ew - nw - left
            tmat = np.array([[1, 0, scale*0.5-0.5+left], [0, 1, scale*0.5-0.5+top], [0, 0, 1]], np.float32)
            amat = np.dot(tmat, smat)
            return amat
        affine = letterbox_warp(src, dst)[:2,:]
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
        return cv2.warpAffine(img, self.affine, tuple(self.dst), flags=cv2.INTER_LINEAR, borderValue=self.borderValue)

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
        w, h = self.dst
        aug_coords = np.column_stack((coords, np.ones(coords.shape[0])))
        coords = np.dot(aug_coords, self.affine.T)
        coords[..., 0] = np.clip(coords[..., 0], 0, w - 1)
        coords[..., 1] = np.clip(coords[..., 1], 0, h - 1)
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply AffineTransform for the image(s).
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the image(s) after applying affine transform.
        """
        return cv2.warpAffine(segmentation, self.affine, tuple(self.dst), flags=cv2.INTER_NEAREST, borderValue=(255,255,255))

    def apply_warp_matrix(self, warp_matrix: np.ndarray) -> np.ndarray:
        C = np.eye(3)
        C[:2,:] = self.affine
        warp_matrix = C @ warp_matrix

        return warp_matrix

class ResizeT(Transform):
    """
    Augmentation Affine
    """

    def __init__(self, src, dst):
        """
        src:(w, h) raw_img size
        dst:(w, h) out img size
        """
        super().__init__()

        Rs = np.eye(3)
        Rs[0, 0] *= dst[0] / src[0]
        Rs[1, 1] *= dst[1] / src[1]

        affine = Rs[:2,:]
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
        return cv2.warpAffine(img, self.affine, tuple(self.dst), flags=cv2.INTER_LINEAR)

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
        w, h = self.dst
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

class AlbumentationsTransform(Transform):
    def __init__(self, aug):
        self.aug = aug
        self.params = aug.get_params()

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_image(self, image):
        self.params = self.prepare_param(image)
        return self.aug.apply(image, **self.params)

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        if hasattr(self.aug,'apply_to_bboxes'):
            try:
                return np.array(self.aug.apply_to_bboxes(box.tolist(), **self.params))
            except AttributeError:
                return box
        return box

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        if hasattr(self.aug,'apply_to_mask'):
            try:
                return self.aug.apply_to_mask(segmentation, **self.params)
            except AttributeError:
                return segmentation
        return segmentation

    def prepare_param(self, image):
        params = self.aug.get_params()
        if self.aug.targets_as_params:
            targets_as_params = {"image": image}
            params_dependent_on_targets = self.aug.get_params_dependent_on_targets(targets_as_params)
            params.update(params_dependent_on_targets)
        params = self.aug.update_params(params, **{"image": image})
        return params

