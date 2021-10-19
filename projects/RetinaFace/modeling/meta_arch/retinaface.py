'''
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-23 17:00:58
@FilePath       : /RetinaFace.detectron2/retinaface/modeling/meta_arch/retinaface.py
@Description    : 
'''

import logging
import math
import numpy as np
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn

from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.structures.keypoints import Keypoints
from ..landmark_regression import Landmark2LandmarkTransform


__all__ = ['RetinaFace']


def permute_all_cls_box_landmark_to_N_HWA_K_and_concat(box_cls, box_delta, landmark_delta, num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    landmark_delta_flattened = [permute_to_N_HWA_K(x, 10) for x in landmark_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    landmark_delta = cat(landmark_delta_flattened, dim=1).view(-1, 10)
    return box_cls, box_delta, landmark_delta


class ssh_block(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        assert out_channel % 4 == 0, "SSH Module required out_channel to be n * 4"
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel // 2)
        )

        self.conv5X5_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
            nn.LeakyReLU(negative_slope=leaky)
        )
        self.conv5X5_2 = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
        )

        self.conv7X7_2 = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
            nn.LeakyReLU(negative_slope=leaky)
        )
        self.conv7x7_3 = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel//4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv3X3 = self.conv3X3(x)

        conv5X5_1 = self.conv5X5_1(x)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = self.relu(out)
        return out


class SSH(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        
        # Add ssh block
        for i, inp_shape in enumerate(input_shape):
            setattr(self, "block{}".format(i), ssh_block(inp_shape.channels, inp_shape.channels))

        # Weight init
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        
        outputs = []
        for i, feature in enumerate(features):
            outputs.append(getattr(self, "block{}".format(i))(feature))
        return outputs
            


class RetinaFaceHead(nn.Module):
    """
    The head used in RetinaFace for object classification, box regression and landmark regression.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels      = input_shape[0].channels
        num_classes      = cfg.MODEL.RETINAFACE.NUM_CLASSES
        prior_prob       = cfg.MODEL.RETINAFACE.PRIOR_PROB
        num_anchors      = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        # Add SSH Module
        self.ssh = SSH(cfg, input_shape)
        
        # Add heads
        cls_score = []
        bbox_pred = []
        # NOTE enable landmark
        landmark_pred = []
        for _ in range(len(input_shape)):
            cls_score.append(nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1, stride=1, padding=0))
            bbox_pred.append(nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1, padding=0))
            landmark_pred.append(nn.Conv2d(in_channels, num_anchors * 10, kernel_size=1, stride=1, padding=0))

        self.cls_score = nn.ModuleList(cls_score)
        self.bbox_pred = nn.ModuleList(bbox_pred)
        self.landmark_pred = nn.ModuleList(landmark_pred)

        # NOTE Initialization

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for cls_score in self.cls_score:
            torch.nn.init.constant_(cls_score.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        landmark_reg = []
        features = self.ssh(features)
        for i, feature in enumerate(features):
            logits.append(self.cls_score[i](feature))
            bbox_reg.append(self.bbox_pred[i](feature))
            landmark_reg.append(self.landmark_pred[i](feature))
        return logits, bbox_reg, landmark_reg


@META_ARCH_REGISTRY.register()
class RetinaFace(nn.Module):
    """
    Implement RetinaFace (arxiv)
    """

    def __init__(self, cfg):
        super().__init__()
        
        # fmt: off
        self.num_classes              = cfg.MODEL.RETINAFACE.NUM_CLASSES
        self.in_features              = cfg.MODEL.RETINAFACE.IN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha         = cfg.MODEL.RETINAFACE.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma         = cfg.MODEL.RETINAFACE.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta      = cfg.MODEL.RETINAFACE.SMOOTH_L1_LOSS_BETA
        self.loc_weight               = cfg.MODEL.RETINAFACE.LOC_WEIGHT
        # Inference parameters:
        self.score_threshold          = cfg.MODEL.RETINAFACE.SCORE_THRESH_TEST
        self.topk_candidates          = cfg.MODEL.RETINAFACE.TOPK_CANDIDATES_TEST
        self.nms_threshold            = cfg.MODEL.RETINAFACE.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # Vis parameters
        self.vis_period               = cfg.VIS_PERIOD
        self.input_format             = cfg.INPUT.FORMAT
        # fmt: on

        self.backbone = build_backbone(cfg)

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = RetinaFaceHead(cfg, feature_shapes)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RETINAFACE.BBOX_REG_WEIGHTS)
        self.landmark2landmark_transform = Landmark2LandmarkTransform(weights=cfg.MODEL.RETINAFACE.LANDMARK_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.MODEL.RETINAFACE.IOU_THRESHOLDS,
            cfg.MODEL.RETINAFACE.IOU_LABELS,
            allow_low_quality_matches=True,
        )

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

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
        img = batched_inputs[image_index]["image"].cpu().numpy()
        assert img.shape[0] == 3, "Images should have 3 channels."
        if self.input_format == "BGR":
            img = img[::-1, :, :]
        img = img.transpose(1, 2, 0)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(
            boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(
            results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)


    def forward(self, batched_inputs):
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
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(
                self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device)
                            for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, landmark_delta = self.head(features)
        anchors = self.anchor_generator(features)

        if self.training:
            gt_classes, gt_anchors_reg_deltas, gt_landmarks_reg_deltas, gt_landmarks_labels = self.get_ground_truth(
                anchors, gt_instances)
            losses = self.losses(
                gt_classes, gt_anchors_reg_deltas, gt_landmarks_reg_deltas, gt_landmarks_labels, box_cls, box_delta, landmark_delta)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        box_cls, box_delta, landmark_delta, anchors, images.image_sizes)
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(
                box_cls, box_delta, landmark_delta, anchors, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, gt_classes, gt_anchors_deltas, gt_landmarks_deltas, gt_landmarks_labels, pred_class_logits, pred_anchor_deltas, pred_landmark_deltas):
        """
        Args:
            For `gt_classes`, `gt_anchors_deltas`, `gt_landmarks_deltas` and `gt_landmarks_labels` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R), (N, R, 4), (N, R, 10) and (N, R), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits`, `pred_anchor_deltas` and `pred_landmark_deltas`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls", "loss_box_reg" and "loss_landmark_reg"
        """
        pred_class_logits, pred_anchor_deltas, pred_landmark_deltas = permute_all_cls_box_landmark_to_N_HWA_K_and_concat(
            pred_class_logits, pred_anchor_deltas, pred_landmark_deltas, self.num_classes
        )  # Shapes: (N x R, K), (N x R, 4) and (N x R, 10), respectively.

        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)
        gt_landmarks_deltas = gt_landmarks_deltas.view(-1, 10)
        gt_landmarks_labels = gt_landmarks_labels.flatten()

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum().item()
        get_event_storage().put_scalar("num_foreground", num_foreground)
        self.loss_normalizer = (
            self.loss_normalizer_momentum * self.loss_normalizer
            + (1 - self.loss_normalizer_momentum) * num_foreground
        )

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, self.loss_normalizer)

        # regression loss
        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas[foreground_idxs],
            gt_anchors_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        ) / max(1, self.loss_normalizer)
        # scale location loss
        loss_box_reg = self.loc_weight * loss_box_reg

        # landmark regression loss
        # NOTE filter in-valid landmarks
        landmark_foreground_idxs = foreground_idxs & (gt_landmarks_labels > 0)
        # NOTE loss_normalizer for landmark may be not consistence with score or bbox
        loss_landmark_reg = smooth_l1_loss(
            pred_landmark_deltas[landmark_foreground_idxs],
            gt_landmarks_deltas[landmark_foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        ) / max(1, self.loss_normalizer)

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg, "loss_landmark_reg": loss_landmark_reg}

    @torch.no_grad()
    def get_ground_truth(self, anchors, targets):
        """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each anchor.
                R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
                Anchors with an IoU with some target higher than the foreground threshold
                are assigned their corresponding label in the [0, K-1] range.
                Anchors whose IoU are below the background threshold are assigned
                the label "K". Anchors whose IoU are between the foreground and background
                thresholds are assigned a label "-1", i.e. ignore.
            gt_anchors_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth box2box transform
                targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                anchor is labeled as foreground.
            gt_landmarks_deltas (Tensor):
                Shape (N, R, 10).
                The last dimension represents ground-truth landmark2landmark transform
                targets (dx1, dy1, dx2, ..., dx5, dy5) that map each anchor to its matched ground-truth landmark.
                The values in the tensor are meaningful only when the corresponding
                anchor is labeled as foreground and values >= 0.
            gt_landmarks_labels (Tensor):
                Shape (N, R)
                "0" means invalid, "1" means foreground.
        """
        gt_classes = []
        gt_anchors_deltas = []
        gt_landmarks_deltas = []
        gt_landmarks_labels = []
        anchors = Boxes.cat(anchors)

        for targets_per_image in targets:
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, anchors)
            gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)

            has_gt = len(targets_per_image) > 0
            if has_gt:
                # ground truth box regression
                matched_gt_boxes = targets_per_image.gt_boxes[gt_matched_idxs]
                matched_gt_landmarks = targets_per_image.gt_keypoints[gt_matched_idxs].tensor
                matched_gt_landmarks, gt_landmarks_labels_i = matched_gt_landmarks[:, :, :2], matched_gt_landmarks[:, :, 2]
                matched_gt_landmarks = matched_gt_landmarks.reshape(matched_gt_landmarks.shape[0], -1)
                gt_landmarks_labels_i = gt_landmarks_labels_i.reshape(gt_landmarks_labels_i.shape[0], -1)
                gt_landmarks_labels_i, _ = torch.min(gt_landmarks_labels_i, dim=1)

                gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                    anchors.tensor, matched_gt_boxes.tensor
                )
                gt_landmarks_reg_deltas_i = self.landmark2landmark_transform.get_deltas(
                    anchors.tensor, matched_gt_landmarks
                )

                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_classes_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_classes_i[anchor_labels == -1] = -1

            else:
                gt_classes_i = torch.zeros_like(
                    gt_matched_idxs) + self.num_classes
                gt_anchors_reg_deltas_i = torch.zeros_like(
                    anchors.tensor)
                _anchors_num = anchors.tensor.shape[0]
                gt_landmarks_reg_deltas_i = torch.zeros(_anchors_num, 10).to(self.device)
                gt_landmarks_labels_i = torch.zeros(_anchors_num).to(self.device)

            gt_classes.append(gt_classes_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)
            gt_landmarks_deltas.append(gt_landmarks_reg_deltas_i)
            gt_landmarks_labels.append(gt_landmarks_labels_i)

        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas), torch.stack(gt_landmarks_deltas), torch.stack(gt_landmarks_labels)

    def inference(self, box_cls, box_delta, landmark_delta, anchors, image_sizes):
        """
        Arguments:
            box_cls, box_delta, landmark_delta: Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        landmark_delta = [permute_to_N_HWA_K(x, 10) for x in landmark_delta]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4 or 10)

        for img_idx, image_size in enumerate(image_sizes):
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_delta]
            landmark_reg_per_image = [landmark_reg_per_level[img_idx] for landmark_reg_per_level in landmark_delta]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, landmark_reg_per_image, anchors, tuple(
                    image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, landmark_delta, anchors, image_size):
        """
        Single-image inference. Return bounding-box and landmark detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            landmark_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 10.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        landmarks_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, landmark_reg_i, anchors_i in zip(box_cls, box_delta, landmark_delta, anchors):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            landmark_reg_i = landmark_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)
            predicted_landmarks = self.landmark2landmark_transform.apply_deltas(landmark_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            landmarks_all.append(predicted_landmarks)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, landmarks_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, landmarks_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all,
                           class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        keypoints_all = landmarks_all[keep].reshape(-1, 5, 2)
        keypoints_all = torch.cat(
            (keypoints_all, 2 * torch.ones(keypoints_all.shape[0], 5, 1).to(self.device)), dim=2)
        result.pred_keypoints = keypoints_all  # Keypoints(keypoints_all)
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility)
        return images
