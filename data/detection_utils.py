'''
Date: 2021-10-19 21:16:26
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-01-22 16:25:59
FilePath: /D2/data/detection_utils.py
'''


# import detectron2.data.transforms as T
import logging
import data.augmentations as T
import albumentations as A

__all__ = [
    "build_transform_gen",
]

def build_transform_gen(cfg, is_train=True):
    """
    Create a list of :class:`Augmentation` from config.

    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger("detectron2.data.detection_utils")
    tfm_gens = []
    if is_train:
        for (aug, args) in cfg.INPUT.TRAIN_PIPELINES:
            if aug == 'AlbumentationsWrapper':
                sub_aug, sub_args = args
                sub_fun = getattr(A, sub_aug)(**sub_args)
                tfm_gens.append(getattr(T, aug)(sub_fun))
            else:
                tfm_gens.append(getattr(T, aug)(**args))
    else:
        for (aug, args) in cfg.INPUT.TEST_PIPELINES:
            tfm_gens.append(getattr(T, aug)(**args))

    logger.info("TransformGens used(update): " + str(tfm_gens))

    return tfm_gens
