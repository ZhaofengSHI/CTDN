


python kd_mlc_ft.py --log_dir ./kd_mlc_visor_hos_logs --prefix visor_hos_final --lamda3 0.1 --max_epoches 30 --dataset visor_hos --pretrained_model ./kd_mlc_egohos_hands_logs/egohos_hands_80_final/checkpoints/kd_mlc_epoch_final.pt
sleep 5s
CUDA_VISIBLE_DEVICES=4,5,6,7 python generate_cams_visor_hos.py --model ./kd_mlc_visor_hos_logs/visor_hos_final/checkpoints/kd_mlc_epoch_final.pt --cam_out_dir ./output_final/visor_hos/cams_visor_hos_final
sleep 5s
python eval_cam_with_crf.py --cam_out_dir ./output_final/visor_hos/cams_visor_hos_final --gt_root /data1/zhaofeng/Ego_seg/VISOR_HOS_VOC/SegmentationClassgray --eval_only
sleep 5s
python eval_cam_with_crf.py --cam_out_dir ./output_final/visor_hos/cams_visor_hos_final --gt_root /data1/zhaofeng/Ego_seg/VISOR_HOS_VOC/SegmentationClassgray --pseudo_mask_save_path ./output_final/visor_hos/pseudo_masks_visor_hos
sleep 5s
python color.py
sleep 5s
python eval_cam.py --cam_out_dir ./output_final/visor_hos/cams_visor_hos_final --gt_root /data1/zhaofeng/Ego_seg/VISOR_HOS_VOC/SegmentationClassgray