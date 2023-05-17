#!/bin/bash

cd /opt/app/yolov7obb/utils/nms_rotated && python3 setup.py build && cp build/lib.linux-x86_64-3.8/nms_rotated_ext.cpython-38-x86_64-linux-gnu.so . && cd /opt/app

. /usr/local/nvm/nvm.sh && jupyter lab --no-browser --port=10000 --ip=0.0.0.0 --NotebookApp.token='' 0<&- &>/dev/null &
tensorboard --logdir runs/train --host=0.0.0.0 0<&- &>/dev/null &

while :
do
	sleep 10
done