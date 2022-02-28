'''
Date: 2021-10-16 17:58:48
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-02-28 21:06:26
FilePath: /D2/data/transforms/transform.py
'''
from typing import Any, Dict, Union, Tuple

import cv2
import numpy as np
import math
import random

from detectron2.data.transforms import (
    Transform,
    HFlipTransform,
    VFlipTransform,
    ResizeTransform
)

__all__ = [
    "AffineTransform",
    "ResizeT",
    "AlbumentationsTransform",
    "NanoDetT"
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



def get_flip_matrix(prob=0.5):
    F = np.eye(3)
    if random.random() < prob:
        F[0, 0] = -1
    return F


def get_perspective_matrix(perspective=0.0):
    """

    :param perspective:
    :return:
    """
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
    return P


def get_rotation_matrix(degree=0.0):
    """

    :param degree:
    :return:
    """
    R = np.eye(3)
    a = random.uniform(-degree, degree)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=1)
    return R


def get_scale_matrix(ratio=(1, 1)):
    """

    :param ratio:
    """
    Scl = np.eye(3)
    scale = random.uniform(*ratio)
    Scl[0, 0] *= scale
    Scl[1, 1] *= scale
    return Scl


def get_stretch_matrix(width_ratio=(1, 1), height_ratio=(1, 1)):
    """

    :param width_ratio:
    :param height_ratio:
    """
    Str = np.eye(3)
    Str[0, 0] *= random.uniform(*width_ratio)
    Str[1, 1] *= random.uniform(*height_ratio)
    return Str


def get_shear_matrix(degree):
    """

    :param degree:
    :return:
    """
    Sh = np.eye(3)
    Sh[0, 1] = math.tan(
        random.uniform(-degree, degree) * math.pi / 180
    )  # x shear (deg)
    Sh[1, 0] = math.tan(
        random.uniform(-degree, degree) * math.pi / 180
    )  # y shear (deg)
    return Sh


def get_translate_matrix(translate, width, height):
    """

    :param translate:
    :return:
    """
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation
    return T


def get_resize_matrix(raw_shape, dst_shape, keep_ratio):
    """
    Get resize matrix for resizing raw img to input size
    :param raw_shape: (width, height) of raw image
    :param dst_shape: (width, height) of input image
    :param keep_ratio: whether keep original ratio
    :return: 3x3 Matrix
    """
    r_w, r_h = raw_shape
    d_w, d_h = dst_shape
    Rs = np.eye(3)
    if keep_ratio:
        C = np.eye(3)
        C[0, 2] = -r_w / 2
        C[1, 2] = -r_h / 2

        if r_w / r_h < d_w / d_h:
            ratio = d_h / r_h
        else:
            ratio = d_w / r_w
        Rs[0, 0] *= ratio
        Rs[1, 1] *= ratio

        T = np.eye(3)
        T[0, 2] = 0.5 * d_w
        T[1, 2] = 0.5 * d_h
        return T @ Rs @ C
    else:
        Rs[0, 0] *= d_w / r_w
        Rs[1, 1] *= d_h / r_h
        return Rs

class NanoDetT(Transform):
    """
    Augmentation Affine
    """

    def __init__(self, src_shape, dst_shape, warp_kwargs, keep_ratio):
        """
        src_shape:(w, h) raw_img size
        dst_shape:(w, h) out img size
        """
        super().__init__()


        width, height = src_shape
        # center
        C = np.eye(3)
        C[0, 2] = -width / 2
        C[1, 2] = -height / 2

        # do not change the order of mat mul
        if "perspective" in warp_kwargs and random.randint(0, 1):
            P = get_perspective_matrix(warp_kwargs["perspective"])
            C = P @ C
        if "scale" in warp_kwargs and random.randint(0, 1):
            Scl = get_scale_matrix(warp_kwargs["scale"])
            C = Scl @ C
        if "stretch" in warp_kwargs and random.randint(0, 1):
            Str = get_stretch_matrix(*warp_kwargs["stretch"])
            C = Str @ C
        if "rotation" in warp_kwargs and random.randint(0, 1):
            R = get_rotation_matrix(warp_kwargs["rotation"])
            C = R @ C
        if "shear" in warp_kwargs and random.randint(0, 1):
            Sh = get_shear_matrix(warp_kwargs["shear"])
            C = Sh @ C
        if "flip" in warp_kwargs:
            F = get_flip_matrix(warp_kwargs["flip"])
            C = F @ C
        if "translate" in warp_kwargs and random.randint(0, 1):
            T = get_translate_matrix(warp_kwargs["translate"], width, height)
        else:
            T = get_translate_matrix(0, width, height)
        M = T @ C
        # M = T @ Sh @ R @ Str @ P @ C
        ResizeM = get_resize_matrix((width, height), dst_shape, keep_ratio)
        M = ResizeM @ M
        affine = M[:2,:]
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
        return cv2.warpAffine(img, self.affine, tuple(self.dst_shape))

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
        w, h = self.dst_shape
        aug_coords = np.column_stack((coords, np.ones(coords.shape[0])))
        coords = np.dot(aug_coords, self.affine.T)
        coords[..., 0] = np.clip(coords[..., 0], 0, w - 1)
        coords[..., 1] = np.clip(coords[..., 1], 0, h - 1)
        return coords

