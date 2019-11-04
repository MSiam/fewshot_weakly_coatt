#!/usr/bin/env python3

"""Training and evaluation entry point."""
import argparse
from train import meta_train
from test_oslsm_setup import test
from common.gen_experiments import load_and_save_params, Namespace

parser = argparse.ArgumentParser()

parser.add_argument('--exp_dir',
                    type=str,
                    help='directory where all the experiment info is stored',
                    default='')


def main(argv=None):
    options = parser.parse_args()
    options = Namespace(load_and_save_params(vars(options), options.exp_dir))
    test(options)


if __name__ == '__main__':
    main()
