'''
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-21 12:54:52
@FilePath       : /RetinaFace.detectron2/retinaface/config/defaults.py
@Description    : 
'''

from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# MobileNets
# ---------------------------------------------------------------------------- #
_C.MODEL.MNET = CN()

# Output features
_C.MODEL.MNET.OUT_FEATURES = ['mob3', 'mob4', 'mob5']
# Width mult
_C.MODEL.MNET.WIDTH_MULT = 1.0

# ---------------------------------------------------------------------------- #
# RetinaFace Head
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINAFACE = CN()

# This is the number of foreground classes.
_C.MODEL.RETINAFACE.NUM_CLASSES = 1

_C.MODEL.RETINAFACE.IN_FEATURES = ["p3", "p4", "p5"]

# IoU overlap ratio [bg, fg] for labeling anchors.
# Anchors with < bg are labeled negative (0)
# Anchors  with >= bg and < fg are ignored (-1)
# Anchors with >= fg are labeled positive (1)
_C.MODEL.RETINAFACE.IOU_THRESHOLDS = [0.2, 0.35]  # [0.4, 0.5]
_C.MODEL.RETINAFACE.IOU_LABELS = [0, -1, 1]

# Prior prob for rare case (i.e. foreground) at the beginning of training.
# This is used to set the bias for the logits layer of the classifier subnet.
# This improves training stability in the case of heavy class imbalance.
_C.MODEL.RETINAFACE.PRIOR_PROB = 0.01

# Inference cls score threshold, only anchors with score > INFERENCE_TH are
# considered for inference (to improve speed)
_C.MODEL.RETINAFACE.SCORE_THRESH_TEST = 0.2  # 0.02
# Widerface dense faces
_C.MODEL.RETINAFACE.TOPK_CANDIDATES_TEST = 2000
_C.MODEL.RETINAFACE.NMS_THRESH_TEST = 0.4

# Weights on (dx, dy, dw, dh) for normalizing RetinaFace anchor regression targets
_C.MODEL.RETINAFACE.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
# Weights on (dx1, dy1, ..., dx5, dy5) for normalizing RetinaFace landmark regression targets
_C.MODEL.RETINAFACE.LANDMARK_REG_WEIGHTS = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0)

# Loss parameters
_C.MODEL.RETINAFACE.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.RETINAFACE.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.RETINAFACE.SMOOTH_L1_LOSS_BETA = 1.0 # 0.1
_C.MODEL.RETINAFACE.LOC_WEIGHT = 2.0
