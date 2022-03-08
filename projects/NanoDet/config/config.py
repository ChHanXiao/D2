'''
Date: 2021-10-17 15:41:24
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-03-08 19:21:32
FilePath: /D2/projects/NanoDet/config/config.py
'''

from detectron2.config import CfgNode as CN

def add_nanodet_config(cfg):
    cfg.MODEL.YML = "yamls/nanodet/yml/legacy_v0.x_configs/nanodet-m.yml"
    cfg.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]
    cfg.MODEL.PIXEL_STD = [57.375, 57.12, 58.395]
    cfg.MODEL_EMA = CN()
    cfg.MODEL_EMA.ENABLED = False
    cfg.MODEL_EMA.DECAY = 0.9998
    cfg.MODEL_EMA.DEVICE = 'cuda'
    cfg.SOLVER.OPTIM = "SGD"
    cfg.SOLVER.BASE_LR = 0.14
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.NESTEROV = True
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.MAX_ITER = 184750
    cfg.SOLVER.WARMUP_METHOD = 'linear'
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.COSINE_PARAM = (1, 0.05)
    cfg.SOLVER.WARMUP_FACTOR = 0.0001
    cfg.SOLVER.DETACH_ITER = -1
    cfg.SOLVER.IMS_PER_BATCH = 192
    cfg.INPUT.SIZE = (320, 320)
    cfg.INPUT.TRAIN_PIPELINES = [
        ("RandomFlip", dict()),
        ("RandomBrightness", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomContrast", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomSaturation", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomLighting", dict(scale=0.1)),
        ("CenterAffine", dict(output_size=(320, 320))), #(w,h)
    ]
    cfg.INPUT.TEST_PIPELINES = [
        ('CenterAffine', dict(output_size=(320, 320))),
        ]
    cfg.INPUT.FORMAT = "BGR"