<!--
 * @Date: 2021-10-19 22:14:27
 * @Author: ChHanXiao
 * @Github: https://github.com/ChHanXiao
 * @LastEditors: ChHanXiao
 * @LastEditTime: 2022-02-26 22:30:10
 * @FilePath: /D2/README.md
-->
基于detectron2目标检测

预训练权重已验证，[模型下载](https://pan.baidu.com/s/1OrXseqRegTJwb8YOBvAdkw)（密码:l4ov）

使用预训练权重eval需关闭MODEL_EMA

<details>
    <summary>nanodet (click to expand)</summary>

    训练：nanodet模型结构和loss计算均为原版迁移，其他策略也基本相同，数据增强有差异，按道理训练指标不会和原版差异太大
    测试：已同步原版，mAP基本无差异

</details>
<details>
    <summary>yolov5 (click to expand)</summary>

    训练：yolov5模型结构和loss计算均为原版迁移，warmup和lr有差异
    测试：mAP有点差异，因为预处理用的warpAffine，原版为resize，在大输入模型中尤为明显

</details>

TODO

 - [ ] yolov5关键点
 - [x] 添加EMA
 - [x] 添加AdamW
 - [x] mosaic数据增强
 - [x] yolox训练流程

更新
 - [2022.02.10] 同步nanodet-plus、yolov5 训练细节
 - [2022.01.21] 添加[nanodet-plus](https://github.com/RangiLyu/nanodet)，~~未加AdamW、未加EMA~~
 - [2021.11.09] 添加mosaic 参考[mmdet](https://github.com/open-mmlab/mmdetection)中yolox实现
 - [2021.11.07] 添加[yolov5_yolox](https://gitee.com/SearchSource/yolov5_yolox)、[yolov5](https://github.com/ultralytics/yolov5)
 - [2021.10.19] 添加[nanodet](https://github.com/RangiLyu/nanodet)

参考

[detectron2](https://github.com/facebookresearch/detectron2)

[nanodet](https://github.com/RangiLyu/nanodet)

[detectron2_ema](https://github.com/xiaohu2015/detectron2_ema)

[yolov5_yolox](https://gitee.com/SearchSource/yolov5_yolox)

[yolov5](https://github.com/ultralytics/yolov5)

