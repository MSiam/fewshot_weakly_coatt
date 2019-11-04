#!/usr/bin/env python3

"""Training and evaluation entry point."""
import argparse
from train import meta_train
from test_oslsm_setup import test
from common.gen_experiments import load_and_save_params, Namespace

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
                    default='val',
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


def main(argv=None):
    options = parser.parse_args()
    if options.split not in ['trainval', 'val', 'test']:
        print('Error in split')

    options = Namespace(load_and_save_params(vars(options), options.exp_dir))
    if options.train:
        meta_train(options)

    if options.test_multi_run:
        test_multi_runs(options)
    else:
        test(options)

if __name__ == '__main__':
    main()
