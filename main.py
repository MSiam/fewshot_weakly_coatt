#!/usr/bin/env python3

"""Training and evaluation entry point."""
import argparse
from train import meta_train
from test_oslsm_setup import test
from test_multi_runs import test_multi_runs
from common.gen_experiments import load_and_save_params, Namespace
import torch
import random
import numpy

parser = argparse.ArgumentParser()

parser.add_argument('-lr',
                    type=float,
                    help='learning rate',
                    default=0.00025)

parser.add_argument('-prob',
                    type=float,
                    help='dropout rate of history mask',
                    default=0.7)


parser.add_argument('-bs',
                    type=int,
                    help='batchsize',
                    default=4)

parser.add_argument('-bs_val',
                    type=int,
                    help='batchsize for val',
                    default=64)

parser.add_argument('-model_type',
                    type=str,
                    help='type of model: vanilla, coatt, coatt_nwe',
                    default='vanilla')

parser.add_argument('-fold',
                    type=int,
                    help='fold',
                    default=1)

parser.add_argument('-ckpt',
                    type=str,
                    help='checkpoint directory',
                    default='checkpoint')

parser.add_argument('-gpu',
                    type=str,
                    help='gpu id to use',
                    default='0')

parser.add_argument('-iter_time',
                    type=int,
                    default=5)

parser.add_argument('-split',
                    type=str,
                    default='trainval',
                    help='trainval: train classes but val imgs during validation, \
                          val: val classes and val imgs during validation, \
                          test: val class and val imgs during fewshot test')

parser.add_argument('-film',
                    type=int,
                    default=0)

parser.add_argument('-data_dir',
                    type=str,
                    help='Data directory')

parser.add_argument('-save_vis',
                    type=str,
                    help='visualization directory to save to',
                    default='')

parser.add_argument('-use_web',
                    type=int,
                    help='flag to use web data or pascal5i for support',
                    default=0)

parser.add_argument('-seed',
                    type=int,
                    default=1337)

parser.add_argument('--exp_dir',
                    type=str,
                    help='directory where all the experiment info is stored',
                    default='')

parser.add_argument('-resume',
                    type=int,
                    help='epoch to resume from if 0 doesnt resume',
                    default=0)

parser.add_argument('-train',
                    type=int,
                    help='flag to use training+testing or solely perform testing',
                    default=1)


parser.add_argument('-gamma_steplr',
                    type=float,
                    help='gamma used in step LR scheduler',
                    default=1.0)

parser.add_argument('-step_steplr',
                    type=int,
                    help='step in step LR scheduler',
                    default=50)

parser.add_argument('-reproducability',
                    type=int,
                    help='flag to ensure reproducability of the results but slower training',
                    default=0)

parser.add_argument('-num_epoch',
                    type=int,
                    help='Number of epochs to train',
                    default=200)

parser.add_argument('-embed_type',
                    type=str,
                    default='word2vec',
                    help='word2vec / fasttext / concat')

parser.add_argument('-test_multi_run',
                    type=int,
                    default=0)

parser.add_argument('-n_shots',
                    type=int,
                    default=1)

parser.add_argument('-dataset_name',
                    type=str,
                    default='pascal',
                    help='Name of dataset: pascal/coco')

parser.add_argument('-backbone',
                    type=str,
                    default='resnet50',
                    help='Backbone for Encoder resnet50/ resnet101')

parser.add_argument('-num_workers',
                    type=int,
                    default=1,
                    help='num workers')

parser.add_argument('-ftune_backbone',
                    type=int,
                    default=0,
                    help='option to finetune the backbone or not')

parser.add_argument('-data_aug',
                    type=int,
                    default=1,
                    help='option to augment data or not')

parser.add_argument('-noval',
                    type=int,
                    default=1,
                    help='Flag to not perform validation and save last ckpt')

parser.add_argument('-multires',
                    type=int,
                    default=0,
                    help='Flag to perform pyramidal coattention')

parser.add_argument('-warm_restarts',
                    type=int,
                    default=-1,
                    help='Epoch for next restart, -1 will disable warm restarts and use normal multistep LR for scheduling')

parser.add_argument('-milestone_length',
                    type=float,
                    help='The length of milestone used by LR scheduler',
                    default=0.2)

def main(argv=None):
    options = parser.parse_args()
    if options.split not in ['trainval', 'val', 'test', 'train']:
        print('Error in split')

#    options = Namespace(load_and_save_params(vars(options), options.exp_dir))

    # To ensure reproducability
    if options.reproducability:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(options.seed)
    torch.cuda.manual_seed(options.seed)
    numpy.random.seed(options.seed)
    random.seed(options.seed)

    if options.train:
        meta_train(options)

    if options.test_multi_run:
        test_multi_runs(options, num_runs=int(options.iter_time))
    else:
        test(options)

if __name__ == '__main__':
    main()
