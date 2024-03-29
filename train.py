from torch.utils import data
import torch.optim as optim
import os.path as osp
from utils import *
import time
import torch.nn.functional as F
import tqdm
import random
from dataset_mask_train import Dataset as Dataset_train
from dataset_mask_val import Dataset as Dataset_val
from coco import create_coco_fewshot
import os
import torch
from models import Res_Deeplab
import torch.nn as nn
import numpy as np
import sys
from torch.optim.lr_scheduler import MultiStepLR
from sgdr_optim import CosineAnnealingWithRestartsLR
from common.torch_utils import SnapshotManager
from tensorboardX import SummaryWriter
from coco import create_coco_fewshot
from test_multi_runs import test_multi_runs

def meta_train(options):

    #Set Variables used in experiment
    data_dir = options.data_dir
    gpu_list = [int(x) for x in options.gpu.split(',')]
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    num_class = 2
    num_epoch = options.num_epoch
    step_steplr = num_epoch / 3.0
    learning_rate = options.lr  # 0.000025#0.00025
    
    if options.dataset_name == 'pascal':
        input_size = (321, 321)
    elif options.dataset_name == 'coco':
        input_size = (417, 417)
        
    batch_size = options.bs
    weight_decay = 0.0005
    momentum = 0.9

    milestones = [int((1.0-3.0*options.milestone_length)*num_epoch),
                  int((1.0-2.0*options.milestone_length)*num_epoch),
                  int((1.0-1.0*options.milestone_length)*num_epoch)]

    # Set Vars used for evalution mIoU
    if options.dataset_name == 'pascal':
        nfold_classes = 5
        nfold_out_classes = 15
    else:
        nfold_classes = 20
        nfold_out_classes = 60

    # Create network.
    model = Res_Deeplab(data_dir=data_dir, num_classes=num_class, model_type=options.model_type,
                        filmed=options.film, embed=options.embed_type, dataset_name=options.dataset_name,
                        backbone=options.backbone, multires_flag=options.multires)

    # load resnet-50 preatrained parameter
    model = load_resnet_param(model, model_name=options.backbone, stop_layer='layer4')
    model = nn.DataParallel(model,[0])

    # disable the  gradients of not optomized layers
    if not options.ftune_backbone:
        turn_off(model, filmed=options.film)

    checkpoint_dir = os.path.join(options.exp_dir, options.ckpt, 'fo=%d'% options.fold)
    check_dir(checkpoint_dir)

    # Create training dataset
    if options.dataset_name == 'pascal':
        dataset = Dataset_train(data_dir=data_dir, fold=options.fold, input_size=input_size, normalize_mean=IMG_MEAN,
                                normalize_std=IMG_STD, prob=options.prob, seed=options.seed, n_shots=options.n_shots,
                                data_crop=options.data_aug)
    else:
        dataset, cat_ids = create_coco_fewshot(data_dir, 'train', input_size=input_size,
                                      n_ways=1, n_shots=1, max_iters=30000, fold=options.fold,
                                      prob=options.prob, seed=options.seed, data_aug=options.data_aug)

    trainloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=options.num_workers)

    # when to log loss for tensorboard
    save_pred_every = len(trainloader) - 1

    # create optimizer
    optimizer = optim.SGD([{'params': get_10x_lr_params(model, options.model_type, options.film, options.ftune_backbone),
                            'lr': 10 * learning_rate}],
                            lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Load Snapshot
    snapshot_manager = SnapshotManager(snapshot_dir=os.path.join(checkpoint_dir, 'snapshot'),
                                       logging_frequency=1, snapshot_frequency=1)
    last_epoch = snapshot_manager.restore(model, optimizer)
    print(f'Loaded epoch {last_epoch}')
    if options.warm_restarts == -1:
        if last_epoch == 0:
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=options.gamma_steplr)
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=options.gamma_steplr, last_epoch=0)
            last_epoch = -1
        else:
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=options.gamma_steplr, last_epoch=last_epoch+1)
    else:
        if last_epoch == 0:
            scheduler = CosineAnnealingWithRestartsLR(optimizer, T_max=options.warm_restarts)
            scheduler = CosineAnnealingWithRestartsLR(optimizer, T_max=options.warm_restarts, last_epoch=0)
            last_epoch = -1
        else:
            scheduler = CosineAnnealingWithRestartsLR(optimizer, T_max=options.warm_restarts, last_epoch=last_epoch+1)

    # Compute mapping used for setting validation mIoU
    mapping = {}
    fold_out_classes = set(range(nfold_classes+nfold_out_classes)) - \
                        set(range(options.fold*nfold_classes, (options.fold+1)*nfold_classes))
    for it, c in enumerate(fold_out_classes):
        if options.dataset_name == 'coco': # In case of MS COCO use category ids its not continuous class numbers
            mapping[cat_ids[c]] = it
        else:
            mapping[c+1] = it

    # Set validation metrics and restore from snapshot
    loss_list = [] # track training loss
    iou_list = [] # track validaiton iou
    highest_iou = 0
    best_iou = 0
    if len(snapshot_manager.losses['training']) > 0:
        loss_list = list(snapshot_manager.losses['training'].values())
    if len(snapshot_manager.losses['validation']) > 0:
        iou_list = list(snapshot_manager.losses['validation'].values())
        highest_iou = np.max(iou_list)

    model.cuda()
    model = model.train()

    # initialize vars for summaries and logging
    tensorboard = SummaryWriter(log_dir = os.path.join(checkpoint_dir, 'tensorboard'))
    tempory_loss = 0  # accumulated loss
    best_epoch=0
    snapshot_manager.enable_time_tracking()

    for epoch in range(last_epoch+1, num_epoch):
        print('Running epoch ', epoch, ' from ', num_epoch)
        print('Epoch:', epoch,'LR:', scheduler.get_lr())
        begin_time = time.time()
        tqdm_gen = tqdm.tqdm(trainloader)

        for i_iter, batch in enumerate(tqdm_gen):
            query_rgb, query_mask,support_rgb, support_mask,history_mask, _, _, sample_class,index= batch

            # Set all input vars to cuda
            query_rgb = (query_rgb).cuda(0)
            support_rgb = (support_rgb).cuda(0)
            support_mask = (support_mask).cuda(0)
            query_mask = (query_mask).cuda(0).long()  # change formation for crossentropy use
            query_mask = query_mask[:, 0, :, :]  # remove the second dim,change formation for crossentropy use
            history_mask=(history_mask).cuda(0)

            # Inference
            optimizer.zero_grad()
            if options.model_type == 'vanilla':
                pred=model(query_rgb, support_rgb, support_mask,history_mask)
            else:
                pred=model(query_rgb, support_rgb, sample_class,history_mask)
            pred_softmax=F.softmax(pred,dim=1).data.cpu()

            #update history mask
            for j in range (support_mask.shape[0]):
                sub_index=index[j]
                dataset.history_mask_list[sub_index]=pred_softmax[j]

            # Upsample prediction
            pred = nn.functional.interpolate(pred,size=input_size, mode='bilinear',align_corners=True)#upsample

            # Compute loss and backpropagate
            loss = loss_calc_v1(pred, query_mask, 0)
            loss.backward()
            optimizer.step()

            tqdm_gen.set_description('e:%d loss = %.4f-:%.4f' % (epoch, loss.item(),highest_iou))

            #save training loss
            tempory_loss += loss.item()
            if i_iter % save_pred_every == 0 and i_iter != 0:
                loss_list.append(tempory_loss / save_pred_every)
                plot_loss(checkpoint_dir, loss_list, save_pred_every)
                np.savetxt(os.path.join(checkpoint_dir, 'loss_history.txt'), np.array(loss_list))
                tempory_loss = 0

        if not options.noval: # Only when validation scheme is allowed
            # ======================evaluate now==================
            with torch.no_grad():
                print ('----Evaluation----')
                model = model.eval()

                best_iou = 0
                initial_seed = options.seed #+ epoch
                for eva_iter in range(options.iter_time):
                    if options.dataset_name == 'pascal':
                        valset = Dataset_val(data_dir=data_dir, fold=options.fold, input_size=input_size,
                                             normalize_mean=IMG_MEAN, normalize_std=IMG_STD,
                                             split=options.split, seed=initial_seed+eva_iter, n_shots=options.n_shots,
                                             data_aug=options.data_aug)
                    else:
                        valset, _ = create_coco_fewshot(data_dir, 'trainval', input_size=input_size,
                                                     n_ways=1, n_shots=1, max_iters=1000, fold=options.fold,
                                                     prob=options.prob, seed=initial_seed+eva_iter,
                                                     data_aug=options.data_aug)

                    valset.history_mask_list=[None] * 1000
                    valloader = data.DataLoader(valset, batch_size=options.bs_val, shuffle=False,
                                                num_workers=options.num_workers,
                                                drop_last=False)

                    all_inter, all_union, all_predict = [0] * nfold_out_classes, [0] * nfold_out_classes, [0] * nfold_out_classes
                    for i_iter, batch in enumerate(valloader):
                        query_rgb, query_mask, support_rgb, support_mask, history_mask, _, _, sample_class, index = batch
                        query_rgb = (query_rgb).cuda(0)
                        support_rgb = (support_rgb).cuda(0)
                        support_mask = (support_mask).cuda(0)
                        query_mask = (query_mask).cuda(0).long()  # change formation for crossentropy use

                        query_mask = query_mask[:, 0, :, :]  # remove the second dim,change formation for crossentropy use
                        history_mask = (history_mask).cuda(0)

                        if options.model_type == 'vanilla':
                            pred = model(query_rgb, support_rgb, support_mask,history_mask)
                        else:
                            pred = model(query_rgb, support_rgb, sample_class,history_mask)
                        pred_softmax = F.softmax(pred, dim=1).data.cpu()

                        # update history mask
                        for j in range(support_mask.shape[0]):
                            sub_index = index[j]
                            valset.history_mask_list[sub_index] = pred_softmax[j]

                        pred = nn.functional.interpolate(pred, size=input_size, mode='bilinear',
                                                             align_corners=True)  #upsample  # upsample

                        _, pred_label = torch.max(pred, 1)
                        inter_list, union_list, _, num_predict_list = get_iou_v1(query_mask, pred_label)
                        for j in range(query_mask.shape[0]):#batch size
                            all_inter[mapping[int(sample_class[j])]] += inter_list[j]
                            all_union[mapping[int(sample_class[j])]] += union_list[j]

                    IOU = []
                    for j in range(nfold_out_classes):
                        if all_union[j] != 0:
                            IOU.append(all_inter[j] / all_union[j])

                    mean_iou = np.mean(IOU)
                    print('IOU:%.4f' % (mean_iou))
                    if mean_iou > best_iou:
                        best_iou = mean_iou
                    else:
                        break

                iou_list.append(best_iou)
                plot_iou(checkpoint_dir, iou_list)
                np.savetxt(os.path.join(checkpoint_dir, 'iou_history.txt'), np.array(iou_list))

                if best_iou>highest_iou:
                    highest_iou = best_iou
                    model = model.eval()
                    torch.save(model.cpu().state_dict(), osp.join(checkpoint_dir, 'model', 'best.pth'))
                    model = model.train()
                    best_epoch = epoch
                    print('A better model is saved')

                print('IOU for this epoch: %.4f' % (best_iou))

                model = model.train()
                model.cuda()

        # Measure time for one epoch
        epoch_time = time.time() - begin_time
        print('best epoch:%d ,iout:%.4f' % (best_epoch, highest_iou))
        print('This epoch takes:', epoch_time, 'second')
        print('still need hour:%.4f' % ((num_epoch - epoch) * epoch_time / 3600))

        # Save epoch snapshot
        training_loss = float(loss_list[-1]) if len(loss_list) > 0 else np.nan
        snapshot_manager.register(iteration=epoch,
                                  training_loss=training_loss,
                                  validation_loss=best_iou,
                                  model=model, optimizer=optimizer)

        # Log epoch metrics
        tensorboard.add_scalar('validation/best_iou', best_iou, epoch)
        tensorboard.add_scalar('training/loss', training_loss, epoch)
        tensorboard.add_scalar('training/learning_rate', scheduler.get_lr(), epoch)
        test_miou = test_multi_runs(options, mode='last', num_runs=1)
        tensorboard.add_scalar('test/mean_iou', test_miou, epoch)

        scheduler.step()

    tensorboard.close()

    if options.noval: # Save last checkpoint in case no validation
        model = model.eval()
        torch.save(model.cpu().state_dict(), osp.join(checkpoint_dir, 'model', 'best.pth'))
        print('A model is saved')

