import torch
import io, os
import math
import numpy as np
import copy
import cv2
from detectron2.config import get_cfg
from config.config import add_nanodet_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from modeling.backbone import *
from modeling import *
from detectron2.utils.visualizer import Visualizer

image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names

def load_model(config_file, model_path=None, device='cpu'):
    cfg = get_cfg()
    add_nanodet_config(cfg)
    cfg.merge_from_file(config_file)
    model = build_model(cfg)
    if not model_path == None:
        DetectionCheckpointer(model).load(model_path)
    model.eval()
    model.to(device)
    return model

def get_cate(config_file):
    cfg = get_cfg()
    add_nanodet_config(cfg)
    cfg.merge_from_file(config_file)
    model_yml_file = cfg.MODEL.YML
    from modeling.util import cfg_s, load_config
    load_config(cfg_s, model_yml_file)
    return cfg_s.class_names


def cv2_letterbox_image_by_warp(img_size, expected_size):
    iw,ih = img_size
    ew,eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    smat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float32)
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left
    tmat = np.array([[1, 0, scale*0.5-0.5+left], [0, 1, scale*0.5-0.5+top], [0, 0, 1]], np.float32)
    amat = np.dot(tmat, smat)
    return amat

def infer_process(
    img_path, 
    dst=(640, 640),
    pad_value=(114, 114, 114),
    rescale=True,
    size_divisibility=32
    ):
    """
    img_path: img path
    dst:(w, h) 
    """
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
    src = img.shape[:2][::-1] #(w,h)
    if rescale:
        dst = dst
    else:
        scale = min(dst[0]/src[0],dst[1]/src[1])
        if src[0]>src[1]: #h<w
            new_w = math.ceil(src[1]*scale/size_divisibility)*size_divisibility
            dst=[int(src[0]*scale), new_w]
        else:
            new_h = math.ceil(src[0]*scale/size_divisibility)*size_divisibility
            dst=[new_h, int(src[1]*scale)]

    affine = cv2_letterbox_image_by_warp(src, dst)

    img_meta=dict()
    img_meta['img'] = img
    img_meta['img_size'] = src
    img_meta['warp_matrix'] = affine
    img_infer = cv2.warpAffine(img[:,:,::-1], affine[:2, :], dst, flags=cv2.INTER_LINEAR, borderValue=pad_value)
    # cv2.imwrite('infer.jpg', img_infer)
    img_meta['input'] = np.transpose(img_infer, (2, 0, 1))
    return img_meta

if __name__ == '__main__':
    cfg_path = 'yamls/nanodet/nanodet-plus-m-1.5x-416.yaml'
    model_path = 'work_dirs/weights/nanodet/nanodet-plus-m-1.5x_416_checkpoint.pth'

    img_dir = 'work_dirs/img'
    savepath = 'work_dirs/result'
    device = "cuda:0"
    show_thres = 0.5
    input_size = (416,416)

    model = load_model(cfg_path, model_path, device)
    CATE_LIST = get_cate(cfg_path)

    if os.path.isdir(img_dir):
        files = get_image_list(img_dir)
    else:
        files = [img_dir]
    files.sort()
    for img_path in files:
        img_info = infer_process(img_path,input_size)
        dummy_input = torch.from_numpy(img_info['input']).to(device)
        inputs = [{
            "image": dummy_input,
            "height": img_info['img_size'][1] ,
            "width": img_info['img_size'][0] ,
            "warp_matrix": img_info['warp_matrix']
            }]
        results = model(inputs)
        for k,processed_results in enumerate(results):
            bbox = processed_results['instances'].pred_boxes.tensor.detach().cpu().numpy()
            cls = processed_results['instances'].pred_classes.detach().cpu().numpy()
            scores = processed_results['instances'].scores.detach().cpu().numpy()

            H, W = processed_results['instances'].image_size
            img_name = os.path.basename(img_path)
            for c,(x1,y1,x2,y2),s in zip(cls,bbox,scores):
                if s < show_thres:
                    continue
                x1 = str(max(0, int(x1)))
                y1 = str(max(0, int(y1)))
                x2 = str(min(W-1, int(x2)))
                y2 = str(min(H-1, int(y2)))
                s = str(round(float(s),2))
                line = ','.join([img_name,s,x1,y1,x2,y2])
                cv2.rectangle(img_info['img'],(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                cv2.putText(img_info['img'], CATE_LIST[c]+':'+s, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(savepath, img_name), img_info['img'])