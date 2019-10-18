#!/usr/bin/python

import os
import time
import argparse
import git

os.environ['LANG'] = 'en_CA.UTF-8'

if __name__ == "__main__":
    exp_description = "testing"
    # This is the 5-way 5-sot configuration
    params = dict(
        fold=0,
        ckpt='testing',
        split='val',
        data_dir='DATA_DIR',
        film=0,
        use_web=0,
        save_vis='VIS_DIR',
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--aidata_home', type=str, default="/home/boris/", help='The path of your home in /mnt/.')

    aidata_home = parser.parse_known_args()[0].aidata_home
    exp_tag = '_'.join(find_variables(params))  # extract variable names
    exp_dir = os.path.join(aidata_home, "experiments_fewshot_webly",
                           "%s_%s_%s" % (time.strftime("%y%m%d_%H%M%S"), exp_tag, exp_description))

    project_path = os.path.join(aidata_home, "fewshotwebly_coatt")

    # This is for the reproducibility purposes
    repo_path = '/mnt' + project_path
    repo = git.Repo(path=repo_path)
    params['commit'] = repo.head.object.hexsha

    borgy_args = [
        "--image=images.borgy.elementai.lan/tensorflow/tensorflow:1.4.1-devel-gpu-py3",
        "-w", "/",
        "-e", "PYTHONPATH=%s" % repo_path,
        "-e", "DATA_PATH=/mnt/datasets/public/",
        "-v", "/mnt/datasets/public/:/mnt/datasets/public/",
        "-v", "/mnt/home/boris/:/mnt/home/boris/",
        "-v", "/mnt/scratch/:/mnt/scratch/",
        "--cpu=2",
        "--gpu=1",
        "--mem=16",
        "--restartable"
    ]

    cmd = os.path.join(repo_path, "main.py")

    gen_experiments_dir(params, exp_dir, exp_description, cmd, blocking=True, borgy_args=borgy_args)
