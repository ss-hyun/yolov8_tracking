#!/bin/bash

# This script to automatically adjust the hyperparameters used for object tracking.
# However, it does not work properly because the benchmark used as a base and the custom model have different classes.
# Script be in need of modification.

RUN="yolov8s-all-batch480-best_change-sort-cfg-test"
OUTPUT_DIR="/home/nextlab/sshyun/yolov8_tracking/custom-model"
YOLO_PATH="/home/nextlab/sshyun/dataset/models"
YOLO_NAME="yolov8s-all-batch480-best"
REID_PATH="/home/nextlab/sshyun/dataset/models"
REID_NAME="fast_reid_R50_IBN"
# REID_NAME="osnet_ain_x1_0_msmt17"

function run() {
    python3 examples/evolve.py \
        --yolo-model "${YOLO_PATH}/${YOLO_NAME}.pt" \
        --tracking-method deepocsort \
        --reid-model "${REID_PATH}/${REID_NAME}.pt" \
        --project $OUTPUT_DIR \
        --name $RUN \
        --exist-ok \
        --imgsz 1280 \
        --benchmark MOT17 \
        --n-trials 100 \
        --classes 2 5 7 \
        --device 0,1
       #  --name $YOLO_NAME \
       #  --agnostic-nms \
       #  --save-vid \
       #  --save-txt \
       #  --save-conf
}

run
