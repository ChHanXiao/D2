import logging
import math
import numpy as np
import torch
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
from detectron2.modeling import build_backbone

from .util.general import non_max_suppression, scale_coords
from .loss import ComputeLoss, ComputeXLoss
from .backbone import YOLO_BACKBONE

__all__ = ["Yolo"]

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
class Yolo(nn.Module):
    """
    Implement Yolo
    """

    @configurable
    def __init__(
        self,
        *,
        model: nn.Module,
        loss,
        num_classes,
        conf_thres,
        iou_thres,
        pixel_mean,
        pixel_std,
        vis_period=0,
        input_format="BGR",
    ):
        super().__init__()

        self.model = model

        self.num_classes = num_classes
        self.single_cls = num_classes == 1
        # Inference Parameters
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss = loss
        # self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        # self.loss_normalizer_momentum = 0.9

    @classmethod
    def from_config(cls, cfg):
        model = YOLO_BACKBONE(cfg)
        head = model.model[-1]
        if model.model_type == 'yolov5':
            loss = ComputeLoss(cfg, head)
        elif model.model_type == 'yolox':
            loss = ComputeXLoss(model)
        else:
            loss = ComputeLoss(cfg, head)

        return{
            "model": model,
            "loss": loss,
            "num_classes": head.nc,
            "conf_thres": cfg.MODEL.YOLO.CONF_THRESH,
            "iou_thres": cfg.MODEL.YOLO.IOU_THRES,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        """
        
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

        images = self.preprocess_image(batched_inputs)
        pred = self.model(images.tensor)

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            losses = self.loss(pred, gt_instances)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        pred, images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            raw_size_ = []
            warp_matrix_ = []
            for input_per_image in batched_inputs:
                height = input_per_image.get("height")
                width = input_per_image.get("width")
                warp_matrix = input_per_image.get("warp_matrix", np.eye(3))
                raw_size_.append((height,width))
                warp_matrix_.append(warp_matrix)
            results = self.process_inference(pred, images.image_sizes, raw_size_, warp_matrix_)
            processed_results = []
            for results_per_image in results:
                processed_results.append({"instances": results_per_image})
            return processed_results

    def process_inference(self, out, image_sizes, raw_size_, warp_matrix_):
        out = non_max_suppression(out, self.conf_thres, self.iou_thres, multi_label=True, agnostic=self.single_cls)
        assert len(out) == len(image_sizes)
        results_all: List[Instances] = []
        # Statistics per image
        for si, (pred, img_size, raw_size, warp_matrix) in enumerate(zip(out, image_sizes, raw_size_, warp_matrix_)):
            if len(pred) == 0:
                continue
            # Predictions
            if self.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            predn =predn.detach().cpu().numpy()
            predn[:, :4] = warp_boxes(predn[:, :4], np.linalg.inv(warp_matrix), raw_size[1], raw_size[0])
            predn = torch.from_numpy(predn).to(pred.device)
            # Predn shape [ndets, 6] of format [xyxy, conf, cls] relative to the input image size
            result = Instances(raw_size)
            result.pred_boxes = Boxes(predn[:, :4])  # TODO: Check if resizing needed
            result.scores = predn[:, 4]
            result.pred_classes = predn[:, 5].int()   # TODO: Check the classes
            results_all.append(result)
        return results_all

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 32)
        return images
