#!/bin/bash


set -e



# egohos hands 80
python kd_mlc_ft.py --log_dir ./kd_mlc_egohos_hands_logs --prefix egohos_hands --dataset egohos_hands
sleep 5s
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_cams_egohos_hands.py --model ./kd_mlc_egohos_hands_logs/egohos_hands/checkpoints/kd_mlc_epoch_final.pt --cam_out_dir ./output_final/egohos_hands/cams_egohos_hands_final
sleep 5s
python eval_cam_with_crf.py --cam_out_dir ./output_final/egohos_hands/cams_egohos_hands_final --image_root /data1/zhaofeng/Ego_seg/EgoHOS/JPEGImages --split_file /data1/zhaofeng/Ego_seg/EgoHOS/ImageSets/Segmentation/train.txt --gt_root /data1/zhaofeng/Ego_seg/EgoHOS/SegmentationClassgray_twohands --eval_only
sleep 5s
python eval_cam_with_crf.py --cam_out_dir ./output_final/egohos_hands/cams_egohos_hands_final --gt_root /data/zhaofeng/Ego_seg/EgoHOS/SegmentationClassgray_twohands --pseudo_mask_save_path ./output_final/egohos_hands/pseudo_masks_egohos_hands
sleep 5s
python color.py
sleep 5s
python eval_cam.py --cam_out_dir ./output_final/egohos_hands/cams_egohos_hands_final --gt_root /data/zhaofeng/Ego_seg/EgoHOS/SegmentationClassgray_twohands



