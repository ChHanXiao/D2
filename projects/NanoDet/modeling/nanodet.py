'''
Author: doumu
Date: 2021-09-27 17:44:05
LastEditTime: 2022-02-28 21:12:00
LastEditors: ChHanXiao
Description: 
FilePath: /D2/projects/NanoDet/modeling/nanodet.py
'''
import logging
import math
import numpy as np
import torch
import copy
from torch import nn, Tensor
from typing import List, Dict, Tuple

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, Conv2d
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.postprocessing import detector_postprocess

from .backbone import build_backbone
from .fpn import build_fpn
from .head import build_head

__all__ = ["NanoDet","NanoDetPlus"]


def warp_boxes(boxes, M, width, height):
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        return xy.astype(np.float32)
    else:
        return boxes


@META_ARCH_REGISTRY.register()
class NanoDet(nn.Module):

    @configurable
    def __init__(
        self, 
        *,
        backbone: nn.Module,
        fpn: nn.Module,
        head: nn.Module,
        pixel_mean,
        pixel_std,
        vis_period=0,
        input_format="BGR",
        ):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head
        self.vis_period = vis_period
        self.input_format = input_format
        self.visthresh = 0.3
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.iter = 0
    
    @classmethod
    def from_config(cls, cfg):
        model_yml_file = cfg.MODEL.YML
        from .util import cfg_s, load_config
        load_config(cfg_s, model_yml_file)

        backbone = build_backbone(cfg_s.model.arch.backbone)
        fpn = build_fpn(cfg_s.model.arch.fpn)
        head = build_head(cfg_s.model.arch.head)
        
        return{
            "backbone": backbone,
            "fpn": fpn,
            "head": head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT,
        }
        
    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        if torch.onnx.is_in_onnx_export():
            return self.forward_(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        x = self.fpn(features)
        preds = self.head(x)
        gt_meta=dict()
        gt_meta['img'] = images.tensor

        if self.training:
            # 转成原loss计算格式
            gt_instances = [x["instances"]for x in batched_inputs]
            targets_boxes = []
            targets_classes = []
            for i, gt_per_image in enumerate(gt_instances):
                boxes=[]
                classes=[]
                if len(gt_per_image) > 0:
                    boxes=gt_per_image.gt_boxes.tensor.clone().numpy()
                    classes=gt_per_image.gt_classes.clone().numpy()
                else:
                    boxes = np.zeros((0, 4), dtype=np.float32)
                    classes = np.array([], dtype=np.int64)
                targets_boxes.append(boxes)
                targets_classes.append(classes)
            gt_meta['gt_bboxes'] = targets_boxes
            gt_meta['gt_labels'] = targets_classes
            loss_states = self.head.loss(preds, gt_meta)
            return loss_states
        else:
            results_infer = self.head.post_process(preds, gt_meta)

            raw_size_ = []
            warp_matrix_ = []
            for input_per_image in batched_inputs:
                height = input_per_image.get("height")
                width = input_per_image.get("width")
                warp_matrix = input_per_image.get("warp_matrix", np.eye(3))
                raw_size_.append((width,height))
                warp_matrix_.append(warp_matrix)

            results = self.process_inference(results_infer, images.image_sizes, raw_size_, warp_matrix_)
            processed_results = []
            for results_per_image in results:
                processed_results.append({"instances": results_per_image})
            return processed_results

    def process_inference(self, out, image_sizes, raw_size_, warp_matrix_):
        assert len(out) == len(image_sizes) == len(raw_size_) == len(warp_matrix_)
        results_all: List[Instances] = []
        # Statistics per image
        for si, (pred, img_size, raw_size, warp_matrix) in enumerate(zip(out, image_sizes, raw_size_, warp_matrix_)):
            det_bboxes, det_labels = pred
            if len(det_bboxes) == 0:
                continue
            det_bboxes = det_bboxes.detach().cpu().numpy()
            predc = det_labels.clone()
            det_bboxes[:, :4] = warp_boxes(det_bboxes[:, :4], np.linalg.inv(warp_matrix), raw_size[0], raw_size[1])
            det_bboxes = torch.from_numpy(det_bboxes).to(det_labels.device)
            # Predictions
            predn = det_bboxes.clone()
            # Predn shape [ndets, 6] of format [xyxy, conf, cls] relative to the input image size
            result = Instances((raw_size[1], raw_size[0]))
            result.pred_boxes = Boxes(predn[:, :4])  # TODO: Check if resizing needed
            result.scores = predn[:, 4]
            result.pred_classes = predc   # TODO: Check the classes
            results_all.append(result)
        return results_all
            
    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 0)
        return images

    @torch.no_grad()
    def forward_(self, batched_inputs):
        features = self.backbone(batched_inputs)
        x = self.fpn(features)
        preds = self.head(x)
        return preds

@META_ARCH_REGISTRY.register()
class NanoDetPlus(NanoDet):
    @configurable
    def __init__(
        self, 
        *,
        backbone: nn.Module,
        fpn: nn.Module,
        aux_head: nn.Module,
        head: nn.Module,
        detach_iter,
        pixel_mean,
        pixel_std,
        vis_period=0,
        input_format="BGR",
        ):
        super(NanoDetPlus, self).__init__(
            backbone=backbone,
            fpn=fpn,
            head=head,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            vis_period=vis_period,
            input_format=input_format,
        )
        self.aux_fpn = copy.deepcopy(self.fpn)
        self.aux_head = aux_head
        self.detach_iter = detach_iter
        self.visthresh = 0.3
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    
    @classmethod
    def from_config(cls, cfg):
        model_yml_file = cfg.MODEL.YML
        from .util import cfg_s, load_config
        load_config(cfg_s, model_yml_file)

        backbone = build_backbone(cfg_s.model.arch.backbone)
        fpn = build_fpn(cfg_s.model.arch.fpn)
        head = build_head(cfg_s.model.arch.head)
        aux_head = build_head(cfg_s.model.arch.aux_head)
        return{
            "backbone": backbone,
            "fpn": fpn,
            "aux_head": aux_head,
            "head": head,
            "detach_iter": cfg.SOLVER.DETACH_ITER,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT,
        }
    def forward(self, batched_inputs):
        # for batched_input in batched_inputs:
        #     images = batched_input["image"]
        #     import cv2
        #     img = images.numpy().transpose(1,2,0)
        #     cv2.imwrite('filename.jpg', img)
        #     print(images.shape)

        # for batched_input in batched_inputs:
        #     images = batched_input["image"]
        #     from detectron2.data import detection_utils as utils
        #     from detectron2.utils.visualizer import Visualizer
        #     import cv2
        #     img = batched_input["image"].permute(1, 2, 0).cpu().detach().numpy()
        #     target_fields = batched_input["instances"].get_fields()
        #     img = utils.convert_image_to_rgb(img, self.input_format)
        #     visualizer = Visualizer(img)
        #     vis = visualizer.overlay_instances(boxes=target_fields.get("gt_boxes", None),)
        #     cv2.imwrite('filename.jpg', vis.get_image()[:, :, ::-1])
        if torch.onnx.is_in_onnx_export():
            return self.forward_(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        fpn_feat = self.fpn(features)
        preds = self.head(fpn_feat)

        gt_meta=dict()
        gt_meta['img'] = images.tensor

        if self.training:
            if self.iter >= self.detach_iter:
                aux_fpn_feat = self.aux_fpn([f.detach() for f in features])
                dual_fpn_feat = (
                    torch.cat([f.detach(), aux_f], dim=1)
                    for f, aux_f in zip(fpn_feat, aux_fpn_feat)
                )
            else:
                aux_fpn_feat = self.aux_fpn(features)
                dual_fpn_feat = (
                    torch.cat([f, aux_f], dim=1) 
                    for f, aux_f in zip(fpn_feat, aux_fpn_feat)
                )
            aux_head_out = self.aux_head(dual_fpn_feat)
            
            # 转成原loss计算格式
            gt_instances = [x["instances"]for x in batched_inputs]
            targets_boxes = []
            targets_classes = []
            for i, gt_per_image in enumerate(gt_instances):
                boxes=[]
                classes=[]
                if len(gt_per_image) > 0:
                    boxes=gt_per_image.gt_boxes.tensor.clone().numpy()
                    classes=gt_per_image.gt_classes.clone().numpy()
                else:
                    boxes = np.zeros((0, 4), dtype=np.float32)
                    classes = np.array([], dtype=np.int64)
                targets_boxes.append(boxes)
                targets_classes.append(classes)
            gt_meta['gt_bboxes'] = targets_boxes
            gt_meta['gt_labels'] = targets_classes
            loss_states = self.head.loss(preds, gt_meta, aux_preds=aux_head_out)

            return loss_states
        else:
            results_infer = self.head.post_process(preds, gt_meta)
            raw_size_ = []
            warp_matrix_ = []
            for input_per_image in batched_inputs:
                height = input_per_image.get("height")
                width = input_per_image.get("width")
                warp_matrix = input_per_image.get("warp_matrix", np.eye(3))
                raw_size_.append((width,height))
                warp_matrix_.append(warp_matrix)

            results = self.process_inference(results_infer, images.image_sizes, raw_size_, warp_matrix_)
            processed_results = []
            for results_per_image in results:
                processed_results.append({"instances": results_per_image})
            return processed_results
