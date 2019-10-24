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
                    help='split of classes to validate upon')

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


def main(argv=None):
  options = parser.parse_args()

  options = Namespace(load_and_save_params(vars(options), options.exp_dir))
  print(options)

  meta_train(options)
  test(options)
  

if __name__ == '__main__':
  main()
