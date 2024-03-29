#!/usr/bin/python

import os
import time
import argparse
import git

from common.gen_experiments import gen_experiments_dir, find_variables

os.environ['LANG'] = 'en_CA.UTF-8'

if __name__ == "__main__":
    exp_description = "coco_1n5shots"
    # This is the 5-way 5-sot configuration
    params = dict(
        fold=[0, 1, 2, 3],
        ckpt='testing',
        split=['trainval'],
        data_dir='/mnt/datasets/public/research/COCO/',
        film=0,
        use_web=0,
        save_vis='',
        model_type=['iter_nwe_coatt', 'coatt', 'nwe', 'nwe_coatt'], 
        seed=[1237, 8773, 9258, 5118, 19783],
        num_epoch=[250],
        milestone_length=[0.05],
        gamma_steplr=0.1,
        bs=[4],
        lr=0.001,
        embed_type=['word2vec'],
        dataset_name=['coco'], # ['pascal', 'coco']
        test_multi_run=1,
        train=1,
        reproducability=0,
        n_shots=[1, 5],
        multires=0,
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

