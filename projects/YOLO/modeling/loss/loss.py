import torch
import torch.nn as nn
import numpy as np
from ..util.general import bbox_iou

from detectron2.config import configurable


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria =
    # FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power
        # for gradient stability

        # TF implementation
        # https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class WingLoss(nn.Module):
    def __init__(self, w=10, e=2):
        super(WingLoss, self).__init__()
        # https://arxiv.org/pdf/1711.06753v4.pdf   Figure 5
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def forward(self, x, t, sigma=1):
        weight = torch.ones_like(t)
        weight[torch.where(t==-1)] = 0
        diff = weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < self.w).float()
        y = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (abs_diff - self.C)
        return y.sum()

class LandmarksLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=1.0):
        super(LandmarksLoss, self).__init__()
        self.loss_fcn = WingLoss()#nn.SmoothL1Loss(reduction='sum')
        self.alpha = alpha

    def forward(self, pred, truel, mask):
        loss = self.loss_fcn(pred*mask, truel*mask)
        return loss / (torch.sum(mask) + 10e-14)

class ComputeLoss(object):
    # Compute losses

    @configurable
    def __init__(self,
                 *,
                 focal_loss_gamma,
                 box_loss_gain,
                 cls_loss_gain,
                 lmk_loss_gain,
                 cls_positive_weight,
                 obj_loss_gain,
                 obj_positive_weight,
                 label_smoothing=0.0,
                 gr,
                 na,
                 nc,
                 nl,
                 kp,
                 anchors,
                 anchor_t,
                 stride,
                 autobalance=False,
                 ):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False
        self.na = na
        self.nc = nc
        self.nl = nl
        self.kp = kp

        self.anchors = anchors
        self.box_loss_gain = box_loss_gain
        self.cls_loss_gain = cls_loss_gain
        self.obj_loss_gain = obj_loss_gain
        self.lmk_loss_gain = lmk_loss_gain
        self.anchor_t = anchor_t

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_positive_weight]))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_positive_weight]))
        landmarks_loss = LandmarksLoss(1.0)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # positive, negative BCE targets
        self.cp, self.cn = smooth_BCE(eps=label_smoothing)

        # Focal loss
        if focal_loss_gamma > 0:
            BCEcls = FocalLoss(BCEcls, focal_loss_gamma)
            BCEobj = FocalLoss(BCEobj, focal_loss_gamma)

        # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(stride).index(16) if autobalance else 0  # stride 16 index

        self.BCEcls, self.BCEobj,self.landmarks_loss, self.gr, self.autobalance = BCEcls, BCEobj,landmarks_loss, gr, autobalance

    @classmethod
    def from_config(cls, cfg, head):
        return{
            "focal_loss_gamma": cfg.MODEL.YOLO.FOCAL_LOSS_GAMMA,
            "box_loss_gain": cfg.MODEL.YOLO.BOX_LOSS_GAIN,
            "cls_loss_gain": cfg.MODEL.YOLO.CLS_LOSS_GAIN,
            "lmk_loss_gain": cfg.MODEL.YOLO.LMK_LOSS_GAIN,
            "cls_positive_weight": cfg.MODEL.YOLO.CLS_POSITIVE_WEIGHT,
            "obj_loss_gain": cfg.MODEL.YOLO.OBJ_LOSS_GAIN,
            "obj_positive_weight": cfg.MODEL.YOLO.OBJ_POSITIVE_WEIGHT,
            "label_smoothing": cfg.MODEL.YOLO.LABEL_SMOOTHING,
            "gr": 1.0,
            "na": head.na,
            "nc": head.nc,
            "nl": head.nl,
            "kp": head.kp,
            "anchors": head.anchors,
            "anchor_t": cfg.MODEL.YOLO.ANCHOR_T,
            "stride": head.stride,
            "autobalance": False,
        }

    def _initialize_ssi(self, stride):
        if self.autobalance:
            self.ssi = list(stride).index(16)

    def __call__(self, p, instances):  # predictions, targets, model is ignored
        device = instances[0].gt_boxes.device
        self.to(device)
        lcls, lbox, lobj, llmk  = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors, tlandmarks, lmks_mask  = self.build_targets(p, instances)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # prediction subset corresponding to targets
                ps = pi[b, a, gj, gi]

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # iou(prediction, target)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) +  self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5+2*self.kp:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5+2*self.kp:], t)  # BCE

                if self.kp>0:
                    plandmarks = ps[:, 5:5+2*self.kp].sigmoid() * 8. - 4.
                    for k in range(self.kp):
                        plandmarks[:, 0+2*k:2+2*k] = plandmarks[:, 0+2*k:2+2*k] * anchors[i]
                    llmk += self.landmarks_loss(plandmarks, tlandmarks[i], lmks_mask[i])

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.box_loss_gain
        lobj *= self.obj_loss_gain
        lcls *= self.cls_loss_gain
        if self.kp>0:
            llmk *= self.lmk_loss_gain
            return {
                "loss_box": lbox,
                "loss_obj": lobj,
                "loss_cls": lcls,
                "loss_lmk": llmk,
            }
        return {
            "loss_box": lbox,
            "loss_obj": lobj,
            "loss_cls": lcls,
        }

    def build_targets(self, p, gt_instances):
        """
        Args:
            p (list[Tensors]): A list of #feature level predictions
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = []
        for i, gt_per_image in enumerate(gt_instances):
            # Convert the boxes to target format of shape [sum(nL per image), 6]
            # where each target entry is [img_index, class, x, y, w, h],
            # x, y, w, h - relative and x, y are centers
            if len(gt_per_image) > 0:
                h, w = gt_per_image.image_size
                boxes = gt_per_image.gt_boxes.tensor.clone()
                boxes[:, 0:2] = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
                boxes[:, 2:4] = (boxes[:, 2:4] - boxes[:, 0:2]) * 2
                boxes[:, ::2] /= float(w)
                boxes[:, 1::2] /= float(h)
                classes = torch.unsqueeze(gt_per_image.gt_classes.clone(), dim=1)

                if self.kp>0:
                    lmks = gt_per_image.gt_keypoints.tensor.clone()
                    lmks[:, :, ::3] /= float(w)
                    lmks[:, :, 1::3] /= float(h)
                    lmks = lmks[:, :, :2].reshape(lmks.shape[0],-1)
                    t = torch.cat([torch.ones_like(classes)*i, classes, boxes, lmks], dim=1)
                else:
                    t = torch.cat([torch.ones_like(classes)*i, classes, boxes], dim=1)
            else:
                t = torch.zeros(0, 5+self.nc+2*self.kp, device=p[0].device)
            targets.append(t)
        targets = torch.cat(targets, 0)

        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, landmarks, lmks_mask = [], [], [], [], [], []

        # normalized to gridspace gain
        gain = torch.ones(7+2*self.kp, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # append anchor indices
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            #landmarks
            gain[6:6+2*self.kp] = torch.tensor(p[i].shape)[[3,2]*self.kp] # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  #
                # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6+2*self.kp].long()  # anchor indices
            # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

            if self.kp>0:
                lks = t[:,6:6+2*self.kp]
                lks_mask = torch.where(lks < 0, torch.full_like(lks, 0.), torch.full_like(lks, 1.0))
                for k in range(self.kp):
                    lks[:, [0+2*k, 1+2*k]] = (lks[:, [0+2*k, 1+2*k]] - gij)
                lmks_mask.append(lks_mask)
                landmarks.append(lks)
        return tcls, tbox, indices, anch, landmarks, lmks_mask

    def to(self, device):
        self.anchors = self.anchors.to(device)
        self.BCEcls.pos_weight = self.BCEcls.pos_weight.to(device)
        self.BCEobj.pos_weight = self.BCEobj.pos_weight.to(device)


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target, xyxy=False):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        if xyxy:
            tl = torch.max(pred[:, :2], target[:, :2])
            br = torch.min(pred[:, 2:], target[:, 2:])
            area_p = torch.prod(pred[:, 2:] - pred[:, :2], 1)
            area_g = torch.prod(target[:, 2:] - target[:, :2], 1)
        else:
            tl = torch.max(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            br = torch.min(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_p = torch.prod(pred[:, 2:], 1)
            area_g = torch.prod(target[:, 2:], 1)

        hw = (br - tl).clamp(min=0)  # [rows, 2]
        area_i = torch.prod(hw, 1)

        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            if xyxy:
                c_tl = torch.min(pred[:, :2], target[:, :2])
                c_br = torch.max(pred[:, 2:], target[:, 2:])
            else:
                c_tl = torch.min(
                    (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
                )
                c_br = torch.max(
                    (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
                )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class ComputeXLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeXLoss, self).__init__()
        # device = next(model.parameters()).device  # get model device
        # h = model.hyp  # hyperparameters

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.det = det

    def __call__(self, p, instances):  # predictions, targets, model
        device = instances[0].gt_boxes.device
        if (not self.det.training) or (len(p) == 0):
            return torch.zeros(1, device=device), torch.zeros(4, device=device)
        targets = []
        for i, gt_per_image in enumerate(instances):
            # Convert the boxes to target format of shape [sum(nL per image), 6]
            # where each target entry is [img_index, class, x, y, w, h],
            # x, y, w, h - relative and x, y are centers
            if len(gt_per_image) > 0:
                boxes = gt_per_image.gt_boxes.tensor.clone()
                h, w = gt_per_image.image_size
                boxes[:, 0:2] = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
                boxes[:, 2:4] = (boxes[:, 2:4] - boxes[:, 0:2]) * 2
                boxes[:, ::2] /= float(w)
                boxes[:, 1::2] /= float(h)
                classes = torch.unsqueeze(gt_per_image.gt_classes.clone(), dim=1)
                t = torch.cat([torch.ones_like(classes)*i, classes, boxes], dim=1)
                targets.append(t)
        targets = torch.cat(targets, 0)

        # self.to(device)

        loss_iou, loss_obj, loss_cls, loss_l1 = self.det.get_losses(*p, targets, dtype=p[0].dtype)
        return {
            "loss_iou": loss_iou,
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
            "loss_l1": loss_l1
        } 
        (loss,
         iou_loss,
         obj_loss,
         cls_loss,
         l1_loss,
         num_fg, ) = self.det.get_losses(*p, targets, dtype=p[0].dtype)
        return loss, torch.hstack((iou_loss, obj_loss, cls_loss, l1_loss)).detach()
