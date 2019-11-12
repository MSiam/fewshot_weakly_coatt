
import argparse
import torch.backends.cudnn as cudnn
import torch
from models import Res_Deeplab
import torch.nn as nn
from dataset_mask_val import Dataset as Dataset_val
from torch.utils import data
import torch.nn.functional as F
from utils import *
import numpy as np
import os
import cv2
from coco import create_coco_fewshot
from common.torch_utils import SnapshotManager

def save(save_dir, support_rgb, support_mask, query_rgb, pred, iter_i):
    for i, (srgb, smask, qrgb, p) in \
            enumerate(zip(support_rgb, support_mask, query_rgb, pred)):
        cv2.imwrite(save_dir+'/sprt/'+'%05d'%(iter_i+i)+'.png', srgb[0].numpy())
        cv2.imwrite(save_dir+'/qry/'+'%05d'%(iter_i+i)+'.png', qrgb.numpy())
        cv2.imwrite(save_dir+'/sprt_lbl/'+'%05d'%(iter_i+i)+'.png', smask[0].cpu().numpy())
        cv2.imwrite(save_dir+'/qry_pred/'+'%05d'%(iter_i+i)+'.png', p.cpu().numpy())

def test_multi_runs(options, mode='best'):
    data_dir = options.data_dir
    torch.backends.cudnn.benchmark = True

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    num_class = 2
    input_size = (500, 500)
    batch_size = options.bs

    cudnn.enabled = True
    # Create network.
    model = Res_Deeplab(data_dir=data_dir, num_classes=num_class, model_type=options.model_type,
                        filmed=options.film, embed=options.embed_type, dataset_name=options.dataset_name)

    #load trained parameter
    checkpoint_dir = os.path.join(options.exp_dir, options.ckpt, 'fo=%d'% options.fold)
    logger = open(os.path.join(checkpoint_dir, 'final_test_miou_%d.txt'%options.n_shots), 'w')
    model=nn.DataParallel(model,[0])
    if mode == 'best':
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'model/best.pth')))
    elif mode == 'last':
        snapshot_manager = SnapshotManager(snapshot_dir=os.path.join(checkpoint_dir, 'snapshot'),
                                           logging_frequency=1, snapshot_frequency=1)
        last_epoch = snapshot_manager.restore(model, optimizer=None)
        print(f'Restored epoch {last_epoch}')

    if options.save_vis != '' and not os.path.exists(options.save_vis):
        os.mkdir(options.save_vis)
        os.mkdir(options.save_vis+'sprt')
        os.mkdir(options.save_vis+'qry')
        os.mkdir(options.save_vis+'qry_pred')
        os.mkdir(options.save_vis+'sprt_lbl')

    if options.dataset_name == 'pascal':
        nfold_classes = 5
        nfold_out_classes = 15
    else:
        nfold_classes = 20
        nfold_out_classes = 60


    with torch.no_grad():
        print ('----Evaluation----')
        model = model.eval()

        initial_seed = options.seed
        eva_iters_means = []
        eva_iters_fgbg_means = []
        for eva_iter in range(5):
            seed = options.seed + eva_iter
            if options.dataset_name == 'pascal':
                inferset = Dataset_val(data_dir=data_dir, fold=options.fold, input_size=input_size, normalize_mean=IMG_MEAN,
                                         normalize_std=IMG_STD, seed=seed, split='test', n_shots=options.n_shots)
            else:
                inferset, cat_ids = create_coco_fewshot(data_dir, 'test', input_size=input_size,
                                              n_ways=1, n_shots=1, max_iters=1000, fold=options.fold,
                                              prob=options.prob, seed=seed)

            valloader = data.DataLoader(inferset, batch_size=options.bs, shuffle=False, num_workers=options.num_workers,
                                        drop_last=False)


            inferset.history_mask_list=[None] * 1000
            best_iou = 0
            all_inter, all_union, all_predict = [0] * nfold_classes, [0] * nfold_classes, [0] * nfold_classes
            all_fgbg_iou = []

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
                    pred, pred_sprt = model(query_rgb, support_rgb, sample_class,history_mask)

                pred_softmax = F.softmax(pred, dim=1).data.cpu()
                pred_sprt_softmax = F.softmax(pred_sprt, dim=1).data.cpu()

                # update history mask
                for j in range(support_mask.shape[0]):
                    sub_index = index[j]
                    inferset.history_mask_list[sub_index] = pred_softmax[j]

                pred = nn.functional.interpolate(pred, size=input_size, mode='bilinear',
                                                     align_corners=True)  #upsample  # upsample
                pred_sprt = nn.functional.interpolate(pred_sprt, size=input_size, mode='bilinear',
                                                     align_corners=True)  #upsample  # upsample
#                plt.figure(1);plt.imshow(pred_softmax.detach().cpu()[0,1]);plt.figure(2);plt.imshow(query_mask[0].cpu());
#                plt.figure(3);plt.imshow(pred_sprt_softmax.detach().cpu()[0,1]);plt.figure(4);plt.imshow(support_mask[0,0,0].cpu());plt.show()

                _, pred_label = torch.max(pred, 1)
                if options.save_vis != '' and eva_iter == 0:
                    save(options.save_vis, sprt_original, support_mask, qry_original,
                        pred_label, i_iter*options.bs)

                inter_list, union_list, _, num_predict_list = get_iou_v1(query_mask, pred_label)
                inter_list_bg, union_list_bg, _, num_predict_list = get_iou_v1(query_mask, pred_label,
                                                                               mode='background')

                iou_fgbg = []
                for j in range(query_mask.shape[0]):#batch size
                    if options.dataset_name == 'pascal':
                        all_inter[sample_class[j] - (options.fold * nfold_classes + 1)] += inter_list[j]
                        all_union[sample_class[j] - (options.fold * nfold_classes + 1)] += union_list[j]
                    else:
                        all_inter[cat_ids.index(sample_class[j]) - (options.fold * nfold_classes)] += inter_list[j]
                        all_union[cat_ids.index(sample_class[j]) - (options.fold * nfold_classes)] += union_list[j]

                    iou_fgbg.append(np.array([inter_list[j] / union_list[j],
                                          inter_list_bg[j] / union_list_bg[j]]).mean())

                all_fgbg_iou.append(np.mean(iou_fgbg))


            IOU = [0] * nfold_classes

            for j in range(nfold_classes):
                IOU[j] = all_inter[j] / all_union[j]

            mean_iou = np.mean(IOU)
            eva_iters_means.append(mean_iou)

            mean_fgbg_iou = np.mean(all_fgbg_iou)
            eva_iters_fgbg_means.append(mean_fgbg_iou)

        mean_iou = np.mean(eva_iters_means)
        fgbg_mean_iou = np.mean(eva_iters_fgbg_means)

        logger.write('IOU:%.4f , FgBg IOU:%.4f\n' % (mean_iou, fgbg_mean_iou))
        logger.close()
        return mean_iou
