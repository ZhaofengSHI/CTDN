#!/bin/bash


set -e

# 80 ep
python kd_mlc_ft.py --log_dir ./kd_mlc_egohands_logs --prefix egohands_final --dataset egohands --data_root /data/zhaofeng/Ego_seg/EgoHands_new --train_split_file /data/zhaofeng/Ego_seg/EgoHands_new/ImageSets/Segmentation/train.txt
sleep 5s
CUDA_VISIBLE_DEVICES=4,5,6,7 python generate_cams_egohands.py --model ./kd_mlc_egohands_logs/egohands_final/checkpoints/kd_mlc_epoch_final.pt --cam_out_dir ./output/egohands/cams_egohands_final
sleep 5s
python eval_cam_with_crf.py --cam_out_dir ./output/egohands/cams_egohands_final --gt_root /data/zhaofeng/Ego_seg/EgoHands_new/SegmentationClassgray --image_root /data/zhaofeng/Ego_seg/EgoHands_new/JPEGImages --eval_only --split_file /data/zhaofeng/Ego_seg/EgoHands_new/ImageSets/Segmentation/train.txt
sleep 5s
python eval_cam_with_crf.py --cam_out_dir ./output/egohands/cams_egohands_final --gt_root /data/zhaofeng/Ego_seg/EgoHands_new/SegmentationClassgray --image_root /data/zhaofeng/Ego_seg/EgoHands_new/JPEGImages --pseudo_mask_save_path ./output/egohands/pseudo_masks_egohands --split_file /data/zhaofeng/Ego_seg/EgoHands_new/ImageSets/Segmentation/train.txt
sleep 5s
python color.py
sleep 5s
python eval_cam.py --cam_out_dir ./output/egohands/cams_egohands_final --gt_root /data/zhaofeng/Ego_seg/EgoHands_new/SegmentationClassgray --split_file /data/zhaofeng/Ego_seg/EgoHands_new/ImageSets/Segmentation/train.txt





