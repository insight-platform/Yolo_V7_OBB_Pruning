#!/bin/bash

SCRIPT_DIR_NAME="$(basename $(dirname "$(realpath "$0")"))"

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