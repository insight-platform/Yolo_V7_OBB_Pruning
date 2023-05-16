#!/bin/bash

SCRIPT_PATH=$(dirname $(realpath "$0"))
SCRIPT_DIR_NAME="$(basename $SCRIPT_PATH)"

python /opt/app/yolov7obb/train_obb.py \
	--workers 8 \
	--device 0 \
	--epochs 350 \
	--global-batch-size 64 \
	--gpu-batch-size 8 \
	--single-cls \
	--plots_debug \
	--data $SCRIPT_PATH/data.yaml \
	--img 640 640 \
	--cfg $SCRIPT_PATH/yolov7.yaml \
	--weights /opt/app/runs/train/fisheye_person_v1.0.0/weights/best_145_prune_1.2_32.pt \
	--name $SCRIPT_DIR_NAME  \
	--hyp $SCRIPT_PATH/hyp.yaml \
	--project /opt/app/runs/train \
	--prune-train \
	--noautoanchor
