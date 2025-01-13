#!/bin/bash


set -e

python kd_mlc_ft.py --log_dir ./kd_mlc_egohos_handobject_logs --prefix egohos_handobject_final --max_epoches 40 --dataset egohos_handobject --pretrained_model ./kd_mlc_egohos_hands_logs/egohos_hands/checkpoints/kd_mlc_epoch_final.pt
sleep 5s
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_cams_egohos_handobject.py --model ./kd_mlc_egohos_handobject_logs/egohos_handobject_final/checkpoints/kd_mlc_epoch_final.pt --cam_out_dir ./output_final/egohos_handobject/cams_egohos_handobject_final
sleep 5s
python eval_cam_with_crf.py --cam_out_dir ./output_final/egohos_handobject/cams_egohos_handobject_final --gt_root /data/zhaofeng/Ego_seg/EgoHOS/SegmentationClassgray_handobject --eval_only
sleep 5s
python eval_cam_with_crf.py --cam_out_dir ./output_final/egohos_handobject/cams_egohos_handobject_final --gt_root /data/zhaofeng/Ego_seg/EgoHOS/SegmentationClassgray_handobject --pseudo_mask_save_path ./output_final/egohos_handobject/pseudo_masks_egohos_handobject
sleep 5s
python color.py
sleep 5s
python eval_cam.py --cam_out_dir ./output_final/egohos_handobject/cams_egohos_handobject_final --gt_root /data/zhaofeng/Ego_seg/EgoHOS/SegmentationClassgray_handobject