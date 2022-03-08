'''
Date: 2022-01-06 21:25:06
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-03-08 19:00:57
FilePath: /D2/projects/engine/defaults.py
'''

import logging
import weakref
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, create_ddp_model, hooks
from detectron2.engine.train_loop import TrainerBase
from detectron2.modeling import build_model
from detectron2.config import CfgNode
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from detectron2.solver.lr_scheduler import LRMultiplier, WarmupParamScheduler
from fvcore.common.param_scheduler import CosineParamScheduler, MultiStepParamScheduler
from fvcore.nn.precise_bn import get_bn_modules
from .train_loop import AMPTrainer_Iter, SimpleTrainer_Iter
from . import model_ema

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
        # add model EMA
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        kwargs.update(model_ema.may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        # add model EMA if enabled
        model_ema.may_build_model_ema(cfg, model)
        return model

    @classmethod
    def build_optimizer(cls, cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )
        if cfg.SOLVER.OPTIM == 'SGD':
            optimizer = maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                                                    params,
                                                    lr=cfg.SOLVER.BASE_LR,
                                                    momentum=cfg.SOLVER.MOMENTUM,
                                                    nesterov=cfg.SOLVER.NESTEROV,
                                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY,)
        elif cfg.SOLVER.OPTIM == 'AdamW':
            optimizer = maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                                                    params,
                                                    lr=cfg.SOLVER.BASE_LR,
                                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY,)  
        else:
            raise ValueError("Unknow optimizer: {}".format(cfg.SOLVER.OPTIM))                           

        return optimizer
        
    @classmethod
    def build_lr_scheduler(cls, cfg: CfgNode, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Build a LR scheduler from config.
        """
        name = cfg.SOLVER.LR_SCHEDULER_NAME

        if name == "WarmupMultiStepLR":
            steps = [x for x in cfg.SOLVER.STEPS if x <= cfg.SOLVER.MAX_ITER]
            if len(steps) != len(cfg.SOLVER.STEPS):
                logger = logging.getLogger(__name__)
                logger.warning(
                    "SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. "
                    "These values will be ignored."
                )
            sched = MultiStepParamScheduler(
                values=[cfg.SOLVER.GAMMA ** k for k in range(len(steps) + 1)],
                milestones=steps,
                num_updates=cfg.SOLVER.MAX_ITER,
            )
        elif name == "WarmupCosineLR":
            sched = CosineParamScheduler(*cfg.SOLVER.COSINE_PARAM)
        else:
            raise ValueError("Unknown LR scheduler: {}".format(name))

        sched = WarmupParamScheduler(
            sched,
            cfg.SOLVER.WARMUP_FACTOR,
            min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
            cfg.SOLVER.WARMUP_METHOD,
        )
        return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)


    @classmethod
    def do_test(cls, cfg, model, evaluators=None):
        # model with ema weights
        logger = logging.getLogger("detectron2")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with model_ema.apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators)
        else:
            results = cls.test(cfg, model, evaluators=evaluators)
        return results

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            model_ema.EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None, # add EMA hook
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.do_test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret