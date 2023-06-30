#!/bin/bash
SOUCRE_ROOT="/app/input/dataset/scenarios/"
MODEL_PATH="/app/input/models/"
OUTPUT_DIR="/app/output"

function run() {
    echo $1
    python track.py \
        --yolo-weights "${MODEL_PATH}yolov8x.pt" \
        --classes 2 5 7 \
        --tracking-method deepocsort \
        --reid-weights "${MODEL_PATH}osnet_ain_x1_0_msmt17.pt" \
        --project $OUTPUT_DIR \
        --source "$1" \
        --vid-stride 5 \
        --imgsz 1280 \
        --save
       #  --name $YOLO_NAME \
       #  --agnostic-nms \
       #  --save-vid \
       #  --save-txt \
       #  --save-conf
}

for (( i=1; i<2; i++ )) do
       SOURCE="${SOUCRE_ROOT}sb${i}.mp4"
       run $SOURCE
done
