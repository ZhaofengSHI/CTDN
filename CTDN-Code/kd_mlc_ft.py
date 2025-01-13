from ast import arg
import dis
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

import numpy as np
import cv2
import random
import torch
import clip
import argparse
import time
from utils import setup_logging
import logging
import torch.nn as nn
from dataset import Dataset_ft
import utils
import torch.nn.functional as F
from poly_lr import PolynomialLRDecay


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root', type=str, default='/data1/zhaofeng/Ego_seg/VISOR_HOS_VOC')
    parser.add_argument('--train_split_file', type=str, default='/data1/zhaofeng/Ego_seg/VISOR_HOS_VOC/ImageSets/Segmentation/train.txt')
    parser.add_argument('--model', type=str, default='/data1/zhaofeng/Ego_seg/ViT-B-16.pt')
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='visor_hos')

    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--max_epoches', type=int, default=80)
    parser.add_argument("--lamda", default=5.0, type=float)
    parser.add_argument("--lamda2", default=2.5, type=float)
    parser.add_argument("--lamda3", default=0.2, type=float) # 0.1 or 0.2

    parser.add_argument("--inter_layer", default=11, type=int)
    parser.add_argument("--margin", default=0.025, type=float)
    parser.add_argument("--crop_scale", default=0.08, type=float)

    parser.add_argument("--template", default='an egocentric origami {}.', type=str)
    
    parser.add_argument('--log_dir', default='./kd_mlc_logs', type=str)
    parser.add_argument('--prefix', type=str, default='kd_mlc')
    args = parser.parse_args()

    if args.dataset == 'egohands':
        from clip_text import new_class_names_egohands as class_names
    # if args.dataset == 'egohos':
    #     from clip_text import new_class_names_egohos as class_names
    if args.dataset == 'egohos_hands':
        from clip_text import new_class_names_egohos_hands as class_names
    if args.dataset == 'egohos_handobject':
        from clip_text import new_class_names_egohos_handobject as class_names
    if args.dataset == 'visor_hos':
        from clip_text import new_class_names_visor_hos as class_names
        
    # train
    train_list = np.loadtxt(args.train_split_file, dtype=str)
    train_list = [x.split('/')[-1] for x in train_list]

    seed=1234
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # ????hash???????????
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    # Logs
    prefix = args.prefix
    # log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    log_dir = os.path.join(args.log_dir, prefix)
    args.log_dir = log_dir

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Arguments: {}'.format(args))

    # model_teacher
    model_teacher, _ = clip.load(args.model, dataset=args.dataset)
    model_teacher.float().cuda().eval()

    for name, param in model_teacher.named_parameters():
        param.requires_grad = False

    model_teacher = nn.DataParallel(model_teacher,device_ids=[0,1,2,3,4,5,6,7])

    # clip_kd
    import clip_kd

    # use pretrain weights?
    if args.pretrained_model: ##
        args.model =  args.pretrained_model ##

    model, _ = clip_kd.load(args.model, template=args.template, inter_layer = args.inter_layer, margin = args.margin, dataset=args.dataset, pretrained = args.pretrained_model)
    model.float().cuda().train()

    for name, param in model.transformer.named_parameters():
        param.requires_grad = False
    
    model = nn.DataParallel(model,device_ids=[0,1,2,3,4,5,6,7])

    # data
    train_dataset = Dataset_ft(args.data_root,train_list,class_names,args.crop_scale,args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batchsize,shuffle=True,num_workers=16,pin_memory=True)

    # losses
    kl_loss = nn.KLDivLoss()
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    # Optimizer
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(model_params, lr=1e-5, betas=(0.9, 0.95)) # (0.9, 0.999) (0.9, 0.95)
    avg_meter_loss = utils.AverageMeter('loss')
    avg_meter_kl_loss = utils.AverageMeter('kl_loss')
    avg_meter_mse_loss = utils.AverageMeter('mse_loss')
    avg_meter_mlc_loss = utils.AverageMeter('mlc_loss')
    avg_meter_contrast_loss = utils.AverageMeter('contrast_loss')

    # poly lr scheduler
    scheduler = PolynomialLRDecay(optimizer,
                                 max_decay_steps=args.max_epoches,
                                 end_learning_rate=1e-7,
                                 power=1.0)

    # Train
    global_step = 0
    max_step = (len(train_dataset) // args.batchsize) * args.max_epoches

    model.train()

    for epoch in range(args.max_epoches):

        for data in train_dataloader:

            image, multi_hot = data
            image = image.cuda()
            multi_hot = multi_hot.cuda()

            with torch.no_grad():
                fgbg_logits_target, v_fea_target = model_teacher(image)

            logits, fgbg_logits, v_fea, contrast_loss = model(image)

            # bce
            logits = F.sigmoid(logits)
            mlc_loss = bce_loss(logits, multi_hot)

            # kl
            fgbg_logits = F.log_softmax(fgbg_logits,dim=1)
            fgbg_logits_target = F.softmax(fgbg_logits_target,dim=1)
            kd_kl_loss = kl_loss(fgbg_logits,fgbg_logits_target.detach()) * args.lamda

            # mse
            kd_mse_loss = mse_loss(v_fea,v_fea_target.detach()) * args.lamda2

            # contrast
            contrast_loss = torch.mean(contrast_loss) * args.lamda3

            loss = mlc_loss + kd_kl_loss + kd_mse_loss + contrast_loss

            avg_meter_kl_loss.add({'kl_loss': kd_kl_loss.item()})
            avg_meter_mlc_loss.add({'mlc_loss':mlc_loss.item()})
            avg_meter_mse_loss.add({'mse_loss':kd_mse_loss.item()})
            avg_meter_contrast_loss.add({'contrast_loss':contrast_loss.item()})
            avg_meter_loss.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if (global_step-1) % 10 == 0:
                train_log = 'Iter:%5d/%5d, KL_Loss:%.4f, MSELoss:%.4f, ContrastLoss:%.4f, MLC_Loss:%.4f, Total_Loss:%.4f, lr: %.7f'%(
                            global_step-1, max_step, avg_meter_kl_loss.pop('kl_loss'), avg_meter_mse_loss.pop('mse_loss'), avg_meter_contrast_loss.pop('contrast_loss'), avg_meter_mlc_loss.pop('mlc_loss'), avg_meter_loss.pop('loss'), optimizer.param_groups[0]['lr'])
                logger.info(train_log)

        scheduler.step() 

    # final 
    model_save_path = os.path.join(checkpoint_dir, 'kd_mlc_epoch_final.pt')
    torch.save(model.module.state_dict(), model_save_path)
    logger.info('save final model to %s'%model_save_path)  