'''
Date: 2021-10-24 08:42:10
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-02-10 18:06:58
FilePath: /D2/projects/YOLO/config/config.py
'''
from detectron2.config import CfgNode as CN

# yolov5 6.0  hyp
# optimizer
    # g0 BatchNorm2d weight (no decay), lr0=0.01
    # g1 weight (with decay), weight decay=0.0005, lr0=0.01
    # g2 bias weight (no decay),, lr0=0.01
    # Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    # SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
# EMA 
    # decay=0.9999

# 其他参考hyp.scratch.yaml，匹配方式按照宽高比阈值anchor_t，iou_t弃用
# 内部参数初始化照抄不修改

def add_yolo_config(cfg):
    cfg.MODEL.YAML = "yamls/yolo/yml/yolov5m.yaml"
    cfg.MODEL.YOLO = CN()
    cfg.MODEL.YOLO.FOCAL_LOSS_GAMMA = 0.0
    cfg.MODEL.YOLO.BOX_LOSS_GAIN = 0.05
    cfg.MODEL.YOLO.CLS_LOSS_GAIN = 0.5
    cfg.MODEL.YOLO.CLS_POSITIVE_WEIGHT = 1.0
    cfg.MODEL.YOLO.OBJ_LOSS_GAIN = 1.0
    cfg.MODEL.YOLO.OBJ_POSITIVE_WEIGHT = 1.0
    cfg.MODEL.YOLO.LABEL_SMOOTHING = 0.0
    cfg.MODEL.YOLO.ANCHOR_T = 4.0
    cfg.MODEL.YOLO.CONF_THRESH = 0.001
    cfg.MODEL.YOLO.IOU_THRES = 0.65
    cfg.MODEL.PIXEL_MEAN = [0.0, 0.0, 0.0]
    cfg.MODEL.PIXEL_STD = [255.0, 255.0, 255.0]
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.MOMENTUM = 0.937
    cfg.SOLVER.NESTEROV = True
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.IMS_PER_BATCH = 64
    cfg.INPUT.SIZE = (640, 640)
    # ===========mosaic==============
    cfg.INPUT.MOSAIC = CN()
    cfg.INPUT.MOSAIC.ENABLED = True
    cfg.INPUT.MOSAIC.PROBABILITY = 0.4
    cfg.INPUT.MOSAIC.IMG_SCALE = (640, 640)
    cfg.INPUT.MOSAIC.CENTER_RATIO = (0.5, 1.5)
    cfg.INPUT.MOSAIC.PAD_VALUE = 114
    # ===============================
    cfg.INPUT.TRAIN_PIPELINES = [
        ("RandomFlip", dict()),
        ("RandomBrightness", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomContrast", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomSaturation", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomLighting", dict(scale=0.1)),
        ("CenterAffine", dict(output_size=(640, 640))),
    ]
    cfg.INPUT.TEST_PIPELINES = [
        ('CenterAffine', dict(output_size=(640, 640))),
        ]
    cfg.INPUT.FORMAT = "RGB"


