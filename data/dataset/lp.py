'''
Date: 2022-03-20 22:57:53
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-03-20 23:03:07
FilePath: /D2/data/dataset/lp.py
'''
 
from detectron2.data.datasets import register_coco_instances

__all__ = [ "register_lp"]

LP_KEYPOINT_NAMES = (
    "left_top", "right_top",
    "left_down", "right_down"
)

LP_KEYPOINT_FLIP_MAP = (
    ("left_top", "right_top"),
    ("left_down", "right_down")
)
lp_metadata = {
    "thing_classes": ["lp"],
    "keypoint_names": LP_KEYPOINT_NAMES,
    "keypoint_flip_map": LP_KEYPOINT_FLIP_MAP,
}
def register_lp(name, json_file, image_root):
    register_coco_instances(name, lp_metadata, json_file, image_root)
