'''
Date: 2022-02-28 21:14:06
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-02-28 21:46:16
FilePath: /D2/projects/NanoDet/export_onnx.py
'''
import torch
import onnx
import os
from detectron2.config import get_cfg
from config.config import add_nanodet_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from modeling import *
from onnxsim import simplify
def load_model(config_file, model_path=None, device='cpu'):
    cfg = get_cfg()
    add_nanodet_config(cfg)
    cfg.merge_from_file(config_file)
    model = build_model(cfg)
    if model_path:
        DetectionCheckpointer(model).load(model_path)
    model.eval()
    model.to(device)
    return model

def export_onnx(model, inputs, output_path):
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            output_path,
            verbose=True,
            keep_initializers_as_inputs=True,
            opset_version=12
        )
if __name__ == '__main__':
    device='cpu'
    cfg_path = 'yamls/nanodet/nanodet-plus-m-320.yaml'
    model_path = 'work_dirs/weights/nanodet/nanodet-plus-m_320_checkpoint.pth'
    save_path = 'work_dirs/export'
    output_file = 'nanodet-plus-m_320.onnx'
    input_size = (1,3,320,320)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_path = os.path.join(save_path,output_file)
    model = load_model(cfg_path, model_path, device)
    inputs = torch.randn(input_size).to(device)
    export_onnx(model, inputs, output_path)

    onnx_model = onnx.load(output_path)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)
    print('finished exporting onnx')