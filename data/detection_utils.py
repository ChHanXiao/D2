'''
Date: 2021-10-19 21:16:26
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2021-10-19 21:31:12
FilePath: /D2/data/detection_utils.py
'''


# import detectron2.data.transforms as T
import logging
import data.augmentations as T

__all__ = [
    "build_transform_gen",
]

def build_transform_gen(cfg, is_train=True):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size)
        )

    logger = logging.getLogger("detectron2.data.detection_utils")
    tfm_gens = []
    if is_train:
        for (aug, args) in cfg.INPUT.TRAIN_PIPELINES:
            if 'CenterAffine' == aug:
                args=dict(output_size=cfg.INPUT.SIZE)
            tfm_gens.append(getattr(T, aug)(**args))
    else:
        for (aug, args) in cfg.INPUT.TEST_PIPELINES:
            if 'CenterAffine' == aug:
                args=dict(output_size=cfg.INPUT.SIZE)
            tfm_gens.append(getattr(T, aug)(**args))

    logger.info("TransformGens used(update): " + str(tfm_gens))

    return tfm_gens
