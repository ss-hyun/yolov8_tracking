#!/bin/bash
SOUCRE_ROOT="/home/nextlab/sshyun/Dataset/scenarios/"

function run() {
    echo $1
    python track.py \
        --yolo-weights yolov8x.pt \
        --classes 2 5 7 \
        --tracking-method deepocsort \
        --reid-weights osnet_ain_x1_0_msmt17.pt \
        --source "$1" \
        --vid-stride 5 \
        --agnostic-nms \
        --save-vid \
        --save-txt \
        --save-conf
}

for (( i=1; i<14; i++ )) do
       SOURCE="${SOUCRE_ROOT}sb${i}.mp4"
       run $SOURCE
done

OUTPUT_DIR="/home/nextlab/sshyun/yolov8_tracking/runs/track/"
MV_DIR="/home/nextlab/sshyun/yolov8_tracking/outputs/"
find $OUTPUT_DIR -name sb*.mp4 -exec cp {} $MV_DIR \;
