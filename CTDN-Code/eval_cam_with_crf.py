import torch
import os
import numpy as np
import torch.nn.functional as F
import joblib
import multiprocessing
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import cv2
from PIL import Image
import argparse


class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }


def crf(n_jobs, is_egohands, is_egohos_hands, is_egohos_handobject, is_visor_hos, cam_out_dir):
    """
    CRF post-processing on pre-computed logits
    """
    
    # Configuration
    torch.set_grad_enabled(False)
    print("# jobs:", n_jobs)

    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

    # Process per sample
    def process(i):

        ori_width = int(640)
        ori_height = int(320)

        # ori_width = int(448)
        # ori_height = int(448)

        image_id = eval_list[i]
        image_path = os.path.join(args.image_root, image_id + '.jpg')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        image = cv2.resize(image,(ori_width,ori_height))##

        label_path = os.path.join(args.gt_root, image_id + '.png')
        gt_label = np.asarray(Image.open(label_path).resize((ori_width,ori_height), resample=Image.NEAREST), dtype=np.int32)##

        # Mean subtraction
        image -= mean_bgr
        # HWC -> CHW
        image = image.transpose(2, 0, 1)

        filename = os.path.join(args.cam_out_dir, image_id + ".npy")
        cam_dict = np.load(filename, allow_pickle=True).item()
        cams = cam_dict['attn_highres']
        bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), 1)
        cams = np.concatenate((bg_score, cams), axis=0)
        prob = cams

        image = image.astype(np.uint8).transpose(1, 2, 0)
        prob = postprocessor(image, prob)

        label = np.argmax(prob, axis=0)
        
        # print(label)
        # print(cam_dict['keys'])
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        # print(keys)
        label = keys[label]
        
        # print(label)

        if not args.eval_only:
            confidence = np.max(prob, axis=0)
            label[confidence < 0.95] = 255
            cv2.imwrite(os.path.join(args.pseudo_mask_save_path, image_id + '.png'), label.astype(np.uint8))
            cv2.waitKey(250)

        return label.astype(np.uint8), gt_label.astype(np.uint8)

    # CRF in multi-process
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
           [joblib.delayed(process)(i) for i in range(len(eval_list))]
    )
    
    if args.eval_only: ########
        preds, gts = zip(*results)

        if is_egohands:
        # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
            score = scores(gts, preds, n_class=5)
            print(score)
            
            metric_path = '/'.join(cam_out_dir.split('/')[:-1])
            metric_path = os.path.join(metric_path,str(args.cam_out_dir.split('/')[-1])+'_metric.txt')
            
            with open(os.path.join(metric_path),'w') as val_f:
                val_f.write(str(score))

        if is_egohos_hands:
            score = scores(gts, preds, n_class=3)
            print(score)

            metric_path = '/'.join(cam_out_dir.split('/')[:-1])
            metric_path = os.path.join(metric_path,str(args.cam_out_dir.split('/')[-1])+'_metric.txt')
            
            with open(os.path.join(metric_path),'w') as val_f:
                val_f.write(str(score))

        if is_egohos_handobject:
            score = scores(gts, preds, n_class=6)
            print(score)

            metric_path = '/'.join(cam_out_dir.split('/')[:-1])
            metric_path = os.path.join(metric_path,str(args.cam_out_dir.split('/')[-1])+'_metric.txt')
            
            with open(os.path.join(metric_path),'w') as val_f:
                val_f.write(str(score))
        
        if is_visor_hos:
            score = scores(gts, preds, n_class=4)
            print(score)

            metric_path = '/'.join(cam_out_dir.split('/')[:-1])
            metric_path = os.path.join(metric_path,str(args.cam_out_dir.split('/')[-1])+'_metric.txt')
            
            with open(os.path.join(metric_path),'w') as val_f:
                val_f.write(str(score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_out_dir", default="./cam_out", type=str)
    parser.add_argument("--pseudo_mask_save_path", default="/home/xxx/code/code48/ablation/usss/voc/val_attn07_crf", type=str)
    parser.add_argument("--split_file", default="/data1/zhaofeng/Ego_seg/VISOR_HOS_VOC/ImageSets/Segmentation/train.txt",
                        type=str)
    parser.add_argument("--gt_root", default="/data1/zhaofeng/Ego_seg/VISOR_HOS_VOC/SegmentationClassgray", type=str)
    parser.add_argument("--image_root", default="/data1/zhaofeng/Ego_seg/VISOR_HOS_VOC/JPEGImages", type=str)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    is_egohands = 'egohands' in args.cam_out_dir
    is_egohos_hands = 'egohos_hands' in args.cam_out_dir
    is_egohos_handobject = 'egohos_handobject' in args.cam_out_dir
    
    is_visor_hos = 'visor_hos' in args.cam_out_dir

    if 'egohands' in args.cam_out_dir:
        eval_list = list(np.loadtxt(args.split_file, dtype=str))
        eval_list = [x.split('/')[-1][:-4] for x in eval_list]
    # if 'egohos' in args.cam_out_dir:
    #     eval_list = list(np.loadtxt(args.split_file, dtype=str))
    #     eval_list = [x.split('/')[-1][:-4] for x in eval_list]
    if 'egohos_hands' in args.cam_out_dir:
        eval_list = list(np.loadtxt(args.split_file, dtype=str))
        eval_list = [x.split('/')[-1][:-4] for x in eval_list]
    if 'egohos_handobject' in args.cam_out_dir:
        eval_list = list(np.loadtxt(args.split_file, dtype=str))
        eval_list = [x.split('/')[-1][:-4] for x in eval_list]
        
    if 'visor_hos' in args.cam_out_dir:
        eval_list = list(np.loadtxt(args.split_file, dtype=str))
        eval_list = [x.split('/')[-1][:-4] for x in eval_list]


    print('{} images to eval'.format(len(eval_list)))

    if not args.eval_only and not os.path.exists(args.pseudo_mask_save_path):
        os.makedirs(args.pseudo_mask_save_path)

    mean_bgr = (104.008, 116.669, 122.675)
    n_jobs =multiprocessing.cpu_count()
    crf(n_jobs, is_egohands, is_egohos_hands, is_egohos_handobject, is_visor_hos, args.cam_out_dir)
