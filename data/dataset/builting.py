'''
Date: 2021-10-16 17:08:14
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2021-10-16 17:56:52
FilePath: /D2/data/dataset/builting.py
'''
from .crowd_human import register_crowd
from .wider_mafa_face import register_face
from .pseudo_label import register_pseudo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
import os

def register_all_crowd(root):
    SPLITS = [
        ("crowd_human_train",
        os.path.join(root, "crowd_human/Annotations/annotation_train.odgt"),
        os.path.join(root, "crowd_human/JPEGImages/"),
        ["person",]),
        ("crowd_human_val",
        os.path.join(root, "crowd_human/Annotations/annotation_val.odgt"),
        os.path.join(root, "crowd_human/JPEGImages/"),
        ["person",]),
        ("crowd_human_face_train",
        os.path.join(root, "crowd_human/Annotations/annotation_train.odgt"),
        os.path.join(root, "crowd_human/JPEGImages/"),
        ["person","face"]),
        ("crowd_human_face_val",
        os.path.join(root, "crowd_human/Annotations/annotation_val.odgt"),
        os.path.join(root, "crowd_human/JPEGImages/"),
        ["person", "face"]),
    ]
    for name, json_file, image_root, class_names in SPLITS:
        register_crowd(name, json_file, image_root, class_names)
        MetadataCatalog.get(name).evaluator_type = "crowd_human"

def register_all_face(root):
    SPLITS = [
        ("face_train",
        os.path.join(root, "wider_face_add_lm_10_10_add_mafa/ImageSets/Main/trainval_all.txt"),
        os.path.join(root, "wider_face_add_lm_10_10_add_mafa/Annotations/"),
        os.path.join(root, "wider_face_add_lm_10_10_add_mafa/JPEGImages/"),
        ["face",]),
        ("face_test",
        os.path.join(root, "wider_face_add_lm_10_10_add_mafa/ImageSets/Main/test.txt"),
        os.path.join(root, "wider_face_add_lm_10_10_add_mafa/Annotations/"),
        os.path.join(root, "wider_face_add_lm_10_10_add_mafa/JPEGImages/"),
        ["face",]),
        ("widerface_crowdhuman_train",
        os.path.join(root, "wider_face_add_lm_10_10_add_mafa/ImageSets/Main/trainval_all.txt"),
        os.path.join(root, "wider_face_add_lm_10_10_add_mafa/Annotations/"),
        os.path.join(root, "wider_face_add_lm_10_10_add_mafa/JPEGImages/"),
        ["person","face"]),
        ("widerface_crowdhuman_test",
        os.path.join(root, "wider_face_add_lm_10_10_add_mafa/ImageSets/Main/test.txt"),
        os.path.join(root, "wider_face_add_lm_10_10_add_mafa/Annotations/"),
        os.path.join(root, "wider_face_add_lm_10_10_add_mafa/JPEGImages/"),
        ["person", "face"]),
    ]
    for name, txt, xml_root, image_root, class_names in SPLITS:
        register_face(name, txt, xml_root, image_root, class_names)
        MetadataCatalog.get(name).evaluator_type = "face"

def register_all_coco_class(root):
    SPLITS = [
        ("coco_person_train",
        {"thing_classes": ["person",] },
        os.path.join(root, "coco/annotations/instances_train2017.json"),
        os.path.join(root, "coco/train2017/")),
        ("coco_person_val",
        {"thing_classes": ["person",] },
        os.path.join(root, "coco/annotations/instances_val2017.json"),
        os.path.join(root, "coco/val2017/")),
        ("coco_car_train",
        {"thing_classes": ["car",] },
        os.path.join(root, "coco/annotations/instances_train2017.json"),
        os.path.join(root, "coco/train2017/")),
        ("coco_car_val",
        {"thing_classes": ["car",] },
        os.path.join(root, "coco/annotations/instances_val2017.json"),
        os.path.join(root, "coco/val2017/")),
        ("coco_person_car_val",
        {"thing_classes": ["person", "car"] },
        os.path.join(root, "coco/annotations/instances_val2017.json"),
        os.path.join(root, "coco/val2017/")),
    ]
    for name, metadata, json_file, image_root in SPLITS:
        register_coco_instances(name, metadata, json_file, image_root)

def register_all_pseudo(root):
    SPLITS = [
        ("pseudo_person_face_train", ["person", "face"]),
    ]
    for name, class_names in SPLITS:
        register_pseudo(name, class_names)
        MetadataCatalog.get(name).evaluator_type = "pseudo_person_face"

def register_widerface(root):
    WIDERFACE_KEYPOINT_NAMES = (
        "left_eye", "right_eye",
        "nose",
        "left_mouth", "right_mouth"
    )

    WIDERFACE_KEYPOINT_FLIP_MAP = (
        ("left_eye", "right_eye"),
        ("left_mouth", "right_mouth")
    )
    
    widerface_metadata = {
        "thing_classes": ["face"],
        "keypoint_names": WIDERFACE_KEYPOINT_NAMES,
        "keypoint_flip_map": WIDERFACE_KEYPOINT_FLIP_MAP,
    }
    SPLITS = [
        ("widerface_train",
        os.path.join(root, "coco/annotations/instances_train2017.json"),
        os.path.join(root, "coco/train2017/")),
    ]
    for name, image_root, json_file in SPLITS:
        register_coco_instances(name, widerface_metadata, json_file, image_root)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_crowd(_root)
register_all_face(_root)
register_all_coco_class(_root)
register_all_pseudo(_root)
register_widerface(_root)