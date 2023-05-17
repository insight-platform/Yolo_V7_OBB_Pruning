# YOLOv7 for Oriented Object Detection

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696).

The code for the implementation of [Yolov5_obb](https://github.com/hukaixuan19970627/yolov5_obb), [Yolov7](https://github.com/WongKinYiu/yolov7).

## Getting Started 
This repo is based on [yolov7](https://github.com/WongKinYiu/yolov7), [yolov5_obb](https://github.com/hukaixuan19970627/yolov5_obb). 


Requirements environment: 
- Nvidia dGPU (Volta, Turing, Ampere, Ada Lovelace)
- Linux OS with driver 525+
- Docker with Compose plugin installed and configured with Nvidia Container Runtime
- git + lfs.

Prerequisites:

Clone the repository with all the necessary code.

``` shell
git clone https://github.com/insight-platform/Yolo_V7_OBB_Pruning.git
```

Move to project folder

```shell
cd Yolo_V7_OBB_Pruning
```

Build docker image

``` shell
DOCKER_BUILDKIT=1 docker build --target yolov7obb -t yolov7obb:1.0 --build-arg UID=`id -u` \
--build-arg GID=`id -g` --build-arg TZ=`cat /etc/timezone` --progress=plain . \
&& docker tag yolov7obb:1.0 yolov7obb:latest
```

Download all files and models to reproduce results

```shell
make download-data
```

A docker compose file has been prepared to make it easy to start the container. You can include additional volumes if you need.
The container automatically runs JupyterLab (http://127.0.0.1:10000) and TensorBoard (http://127.0.0.1:6006) to track learning metrics.

Start container

``` shell
docker-compose up -d
```

Stop container

```Shell
docker-compose down
```


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

