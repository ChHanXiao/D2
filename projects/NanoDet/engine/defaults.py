'''
Date: 2022-01-06 21:25:06
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-01-21 21:23:47
FilePath: /D2/projects/NanoDet/engine/defaults.py
'''

import logging
import weakref
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import TrainerBase
from .train_loop import AMPTrainer_Iter, SimpleTrainer_Iter

class DefaultTrainer_Iter(DefaultTrainer, TrainerBase):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        TrainerBase.__init__(self)

        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer_Iter if cfg.SOLVER.AMP.ENABLED else SimpleTrainer_Iter)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
