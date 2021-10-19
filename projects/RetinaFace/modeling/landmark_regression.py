'''
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-23 20:49:03
@FilePath       : /RetinaFace.detectron2/retinaface/modeling/landmark_regression.py
@Description    : 
'''

import math
from typing import Tuple
import torch


__all__ = ["Landmark2LandmarkTransform"]


@torch.jit.script
class Landmark2LandmarkTransform(object):
    """
    The transformation is parameterized by 10 deltas: (dx1, dy1, ..., dx5, dy5). 
    The transformation shifts a box's center by the offset (dx * width, dy * height).
    """

    def __init__(
        self, weights: Tuple[float, float, float, float, float, float, float, float, float, float]):
        """
        Args:
            weights (10-element tuple): Scaling factors that are applied to the
                (dx1, dy1, ..., dx5, dy5) deltas.
        """
        self.weights = weights

    def get_deltas(self, src_boxes, target_landmarks):
        """
        Get landmark regression transformation deltas (dx1, dy1, ..., dx5, dy5) that can be used
        to transform the `src_boxes` into the `target_landmarks`. That is, the relation
        ``target_landmarks == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_landmarks (Tensor): target of the transformation, e.g., ground-truth
                landmarks.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_landmarks, torch.Tensor), type(target_landmarks)

        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

        src_xy = torch.stack((src_ctr_x, src_ctr_y), dim=1)
        src_xy = src_xy.repeat([1, 5])
        src_wh = torch.stack((src_widths, src_heights), dim=1)
        src_wh = src_wh.repeat([1, 5])

        weights = torch.as_tensor(self.weights).to(src_wh.dtype).to(src_wh.device)
        deltas = weights * (target_landmarks - src_xy) / src_wh

        assert (src_widths > 0).all().item(
        ), "Input boxes to Box2BoxTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx1, dy1, ..., dx5, dy5) to `landmarks`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*10), where k >= 1.
                deltas[i] represents k potentially different class-specific
                landmark transformations for the single landmark landmarks[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        whs = torch.stack((widths, heights), dim=1)
        whs = whs.repeat([1, deltas.shape[1] // 2])

        ctr_xys = torch.stack((ctr_x, ctr_y), dim=1)
        ctr_xys = ctr_xys.repeat([1, deltas.shape[1] // 2])

        weights = torch.as_tensor(self.weights).to(whs.dtype).to(whs.device)
        dxys = deltas / weights
        dxys = dxys * whs

        pred_landmarks = ctr_xys + dxys
        return pred_landmarks
