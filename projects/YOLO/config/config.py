'''
Date: 2021-10-24 08:42:10
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2021-11-09 21:21:38
FilePath: /D2/projects/YOLO/config/config.py
'''
from detectron2.config import CfgNode as CN

def add_yolo_config(cfg):
    cfg.MODEL.YAML = "yamls/yolo/yml/yolov5m.yaml"
    cfg.MODEL.YOLO = CN()
    cfg.MODEL.YOLO.FOCAL_LOSS_GAMMA = 0.0
    cfg.MODEL.YOLO.BOX_LOSS_GAIN = 0.05
    cfg.MODEL.YOLO.CLS_LOSS_GAIN = 0.3
    cfg.MODEL.YOLO.CLS_POSITIVE_WEIGHT = 1.0
    cfg.MODEL.YOLO.OBJ_LOSS_GAIN = 0.7
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
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
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
    cfg.INPUT.CROP.TYPE = False
    cfg.INPUT.TRAIN_PIPELINES = [
        ("RandomFlip", dict()),
        ("RandomBrightness", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomContrast", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomSaturation", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomLighting", dict(scale=0.1)),
        ("CenterAffine", dict()),
    ]
    cfg.INPUT.TEST_PIPELINES = [
        ('CenterAffine', dict()),
        ]
    cfg.INPUT.FORMAT = "RGB"
    cfg.TEST.AUG.SIZE = 640


