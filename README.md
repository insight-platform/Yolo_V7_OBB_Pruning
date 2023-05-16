# YOLOv7 for Oriented Object Detection

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696).

The code for the implementation of [Yolov5_obb](https://github.com/hukaixuan19970627/yolov5_obb), [Yolov7](https://github.com/WongKinYiu/yolov7).

## Getting Started 
This repo is based on [yolov7](https://github.com/WongKinYiu/yolov7), [yolov5_obb](https://github.com/hukaixuan19970627/yolov5_obb). 
Please see [yolov5_obb](https://github.com/hukaixuan19970627/yolov5_obb) for data preparation.

## Training (Not yet train aux model)

Single GPU training

``` shell
# train yolov7 models
python ./yolov7obb/train_obb.py \
    --workers 8 \
    --device 0 \
    --epochs 600 \
    --batch-size 8 \
    --single-cls \
    --data /opt/app/experiments/base_tram_v1.0.0/data.yaml \
    --img 640 640 \
    --cfg /opt/app/experiments/base_tram_v1.0.0/yolov7.yaml \
    --weights /opt/app/weights/yolov7.pt \
    --name yolov7model_tmp \
    --hyp /opt/app/experiments/base_tram_v1.0.0/hyp.yaml \
    --project /opt/app/runs/train \
    --noautoanchor
```

