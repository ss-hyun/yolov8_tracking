#!/bin/bash
# SOUCRE_ROOT="/app/input/dataset/scenarios/"
# MODEL_PATH="/app/input/models/"
# OUTPUT_DIR="/app/outputs"
RUN="yolov8s-all-batch480-best_change-sort-cfg-test"
OUTPUT_DIR="/home/nextlab/sshyun/yolov8_tracking/custom-model"
YOLO_PATH="/home/nextlab/sshyun/dataset/models"
YOLO_NAME="yolov8s-all-batch480-best"
REID_PATH="/home/nextlab/sshyun/dataset/models"
REID_NAME="fast_reid_R50_IBN"
# REID_NAME="osnet_ain_x1_0_msmt17"

function run() {
    echo $1
    python3 examples/track.py \
        --yolo-model "${YOLO_PATH}/${YOLO_NAME}.pt" \
        --tracking-method deepocsort \
        --reid-model "${REID_PATH}/${REID_NAME}.pt" \
        --project $OUTPUT_DIR \
        --name $RUN \
        --exist-ok \
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

run /home/nextlab/sshyun/dataset/scenarios/basic/sb6.mp4
# run /home/nextlab/sshyun/dataset/scenarios/unusual/su2.mp4
run /home/nextlab/sshyun/dataset/scenarios/unusual/su12.mp4
exit

# SOUCRE_ROOT="/home/nextlab/sshyun/dataset/scenarios/basic/"
# for (( i=1; i<14; i++ )) do
#        SOURCE="${SOUCRE_ROOT}sb${i}.mp4"
#        run $SOURCE
# done

SOUCRE_ROOT="/home/nextlab/sshyun/dataset/scenarios/unusual/"

for (( i=1; i<18; i++ )) do
       SOURCE="${SOUCRE_ROOT}su${i}.mp4"
       run $SOURCE
done
