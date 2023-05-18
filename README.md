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
make docker-image-build-notebook
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

Download all pretrained models and experiments files

```shell
make download-data 
```

## Preparing dataset

Open JupyterLab (http://127.0.0.1:10000) and run [prepare_datasets.ipynb](notebook%2Fprepare_datasets.ipynb)

## Training 

Connect to docker container

```shell
docker exec -it yolo_v7_obb_pruning_yolov7obb_1 /bin/bash
```

Single GPU training

``` shell
# train yolov7 models
python /opt/app/yolov7obb/train_obb.py \
	--workers 8 \
	--device 0 \
	--epochs 600 \
	--global-batch-size 64 \
	--gpu-batch-size 8 \
	--single-cls \
	--plots_debug \
	--data /opt/app/experiments/fisheye_person_v1.0.0/data.yaml \
	--img 640 640 \
	--cfg /opt/app/experiments/fisheye_person_v1.0.0/yolov7.yaml \
	--weights /opt/app/weights/yolov7.pt \
	--name $SCRIPT_DIR_NAME \
	--hyp /opt/app/experiments/fisheye_person_v1.0.0/hyp.yaml \
	--project /opt/app/runs/train \
	--sparsity \
	--noautoanchor
```

## Pruning model

```shell
python ./yolov7obb/pruning.py \
    --weights ./runs/train/fisheye_person_v1.0.0/weights/best_145.pt \
    --weights-ref ./runs/train/fisheye_person_v1.0.0/weights/init.pt \
    -o 1.2 \
    -r 32
```

## Exporting model to onnx

```shell
python ./yolov7obb/export.py  \
   --weights /opt/app/runs/train/fisheye_person_v1.0.0/weights/best_145.pt  \
   --img-size 640 640 \
   --batch-size 1 \
   --onnx \
   --grid \
   --end2end \
   --simplify \
   --fp16
```

## Inference and video generation

You can use notebook [predict_video.ipynb](notebook%2Fpredict_video.ipynb) 
to predict on video and generate video with bounding boxes.



