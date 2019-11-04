#!/usr/bin/python

import os
import time
import argparse
import git

from common.gen_experiments import gen_experiments_dir, find_variables

os.environ['LANG'] = 'en_CA.UTF-8'

if __name__ == "__main__":
#     exp_description = "adam_optimizer_and_clipping"
    # This is the 5-way 5-sot configuration
#     params = dict(
#         fold=[0],
#         ckpt='testing',
#         split=['train'],
#         data_dir='/mnt/datasets/public/research/pascal/VOCdevkit/VOC2012/',
#         film=0,
#         use_web=0,
#         save_vis='VIS_DIR',
#         model_type=['nwe', 'iter_nwe_coatt'],
#         train=1,
#         seed=[1337],
#         num_epoch=[120, 180],
#         gamma_steplr=[1.0, 0.5],
#         bs=4,
#         optimizer=['adam'],
#         clip_gradient_norm=[5.0],
#         lr=[2e-6, 1e-6],
#     )
    
    exp_description = "sgd_train_validaton"
    params = dict(
        fold=[0, 1, 2, 3],
        ckpt='testing',
        split=['train', 'val'],
        data_dir='/mnt/datasets/public/research/pascal/VOCdevkit/VOC2012/',
        film=0,
        use_web=0,
        save_vis='VIS_DIR',
        model_type=['nwe', 'iter_nwe_coatt'],
        train=1,
        seed=[1337],
        num_epoch=200,
        gamma_steplr=1.0,
        bs=4,
        optimizer='sgd',
        clip_gradient_norm=5000,
        lr=[0.00025, 0.001],
    )


    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--aidata_home', type=str, default="/scratch/boris/projects/", help='The path of your home in /mnt/.')

    aidata_home = parser.parse_known_args()[0].aidata_home
    exp_tag = '_'.join(find_variables(params))  # extract variable names
    project_path = os.path.join(aidata_home, "fewshotwebly_coatt")
    exp_dir = os.path.join(project_path, "experiments",
                           "%s_%s_%s" % (time.strftime("%y%m%d_%H%M%S"), exp_tag, exp_description))



    # This is for the reproducibility purposes
    repo_path = '/mnt' + project_path
    repo = git.Repo(path=repo_path)
    params['commit'] = repo.head.object.hexsha

    borgy_args = [
        "--image=images.borgy.elementai.net/fewshot_webly:boris",
        "-w", "/",
        "-e", "PYTHONPATH=%s" % repo_path,
        "-e", "DATA_PATH=/mnt/datasets/public/",
        "-v", "/mnt/datasets/public/:/mnt/datasets/public/",
        "-v", "/mnt/scratch/:/mnt/scratch/",
        "--cpu=2",
        "--gpu=1",
        "--mem=16",
        "--restartable"
    ]

    cmd = os.path.join(repo_path, "main.py")

    gen_experiments_dir(params, exp_dir, exp_description, cmd, blocking=True, borgy_args=borgy_args)

