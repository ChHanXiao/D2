'''
Date: 2021-10-17 10:56:07
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-04-10 22:11:32
FilePath: /D2/projects/NanoDet/train_net.py
'''
"""
Yolo Training script
This script is a simplified version of the script in detectron2/tools
"""

from pathlib import Path
import torch
import os
import sys
sys.path.append(os.getcwd())
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    # DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch
)
from detectron2.data import (
    DatasetMapper,
    build_detection_train_loader,
    build_detection_test_loader
)
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg

from modeling import *
from config.config import add_nanodet_config
from data.dataset_mapper import BaseDatasetMapper
from projects.engine.defaults import DefaultTrainer_Iter
from projects.engine import model_ema

class Trainer(DefaultTrainer_Iter):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = Path(cfg.OUTPUT_DIR) / "inference"
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = BaseDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = BaseDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


def setup(args):
    cfg = get_cfg()
    model_ema.add_model_ema_configs(cfg)
    add_nanodet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
