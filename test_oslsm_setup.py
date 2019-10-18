
import argparse
import torch.backends.cudnn as cudnn
import torch
from models import Res_Deeplab
import torch.nn as nn
from dataset_mask_val import OSLSMSetupDataset, WebSetupDataset
from torch.utils import data
import torch.nn.functional as F
from utils import *
import numpy as np
import os
import cv2

def save(save_dir, support_rgb, support_mask, query_rgb, pred, iter_i):
    for i, (srgb, smask, qrgb, p) in \
            enumerate(zip(support_rgb, support_mask, query_rgb, pred)):
        cv2.imwrite(save_dir+'/sprt/'+'%05d'%(iter_i+i)+'.png', srgb.numpy())
        cv2.imwrite(save_dir+'/qry/'+'%05d'%(iter_i+i)+'.png', qrgb.numpy())
        cv2.imwrite(save_dir+'/sprt_lbl/'+'%05d'%(iter_i+i)+'.png', smask[0].cpu().numpy())
        cv2.imwrite(save_dir+'/qry_pred/'+'%05d'%(iter_i+i)+'.png', p.cpu().numpy())

def test(options):
    data_dir = options.data_dir
    torch.backends.cudnn.benchmark = True

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    num_class = 2
    input_size = (500, 500)
    batch_size = options.bs

    cudnn.enabled = True
    # Create network.
    model = Res_Deeplab(num_classes=num_class, model_type=options.model_type, filmed=options.film)

    #load trained parameter
    checkpoint_dir = options.ckpt+'/fo=%d/'% options.fold
    logger = open(checkpoint_dir+'final_test_miou.txt', 'r')
    model=nn.DataParallel(model,[0])
    model.load_state_dict(torch.load(checkpoint_dir+'model/best.pth'))

    if options.use_web:
        Dataset_val = WebSetupDataset
    else:
        Dataset_val = OSLSMSetupDataset

    inferset = Dataset_val(data_dir=data_dir, fold=options.fold, input_size=input_size, normalize_mean=IMG_MEAN,
                             normalize_std=IMG_STD, seed=1386)
    valloader = data.DataLoader(inferset, batch_size=options.bs, shuffle=False, num_workers=4,
                                drop_last=False)

    if options.save_vis != '' and not os.path.exists(options.save_vis):
        os.mkdir(options.save_vis)
        os.mkdir(options.save_vis+'sprt')
        os.mkdir(options.save_vis+'qry')
        os.mkdir(options.save_vis+'qry_pred')
        os.mkdir(options.save_vis+'sprt_lbl')

    with torch.no_grad():
        print ('----Evaluation----')
        model = model.eval()

        inferset.history_mask_list=[None] * 1000
        best_iou = 0
        all_inter, all_union, all_predict = [0] * 5, [0] * 5, [0] * 5
        for i_iter, batch in enumerate(valloader):
            print('Iteration ', i_iter)
            query_rgb, query_mask, support_rgb, support_mask, history_mask, sprt_original, qry_original, \
                    sample_class, index = batch
            query_rgb = (query_rgb).cuda(0)
            support_rgb = (support_rgb).cuda(0)
            support_mask = (support_mask).cuda(0)
            query_mask = (query_mask).cuda(0).long()  # change formation for crossentropy use

            query_mask = query_mask[:, 0, :, :]  # remove the second dim,change formation for crossentropy use
            history_mask = (history_mask).cuda(0)

            if options.model_type == 'vanilla':
                pred = model(query_rgb, support_rgb, support_mask,history_mask)
            else:
                pred=model(query_rgb, support_rgb, sample_class,history_mask)

            pred_softmax = F.softmax(pred, dim=1).data.cpu()

            # update history mask
            for j in range(support_mask.shape[0]):
                sub_index = index[j]
                inferset.history_mask_list[sub_index] = pred_softmax[j]

                pred = nn.functional.interpolate(pred, size=input_size, mode='bilinear',
                                                 align_corners=True)  #upsample  # upsample

            _, pred_label = torch.max(pred, 1)
            if options.save_vis != '':
                save(options.save_vis, sprt_original, support_mask, qry_original,
                    pred_label, i_iter*options.bs)

            inter_list, union_list, _, num_predict_list = get_iou_v1(query_mask, pred_label)
            for j in range(query_mask.shape[0]):#batch size
                all_inter[sample_class[j] - (options.fold * 5 + 1)] += inter_list[j]
                all_union[sample_class[j] - (options.fold * 5 + 1)] += union_list[j]

        IOU = [0] * 5

        for j in range(5):
            IOU[j] = all_inter[j] / all_union[j]

        mean_iou = np.mean(IOU)
        logger.write('IOU:%.4f\n' % (mean_iou))
        logger.close()
