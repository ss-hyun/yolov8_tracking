#!/bin/bash
SOUCRE="/home/nextlab/sshyun/Dataset/scenarios/sb1.mp4"

python track.py \
    --yolo-weights yolov8x.pt \
    --classes 2 5 7 \
    --tracking-method deepocsort \
    --reid-weights osnet_ain_x1_0_msmt17.pt \
    --source $SOUCRE \
    --vid-stride 5 \
    --save-vid \
    --save-txt