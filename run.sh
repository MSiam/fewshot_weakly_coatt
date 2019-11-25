#!/bin/bash
#SBATCH --array=1
#SBATCH --nodes=1
#SBATCH --job-name=fewshot_webly
#SBATCH --gres=gpu:1
#SBATCH --mem=5000M
#SBATCH --time=1-14:00 #d-hh:mm
#SBATCH --output=outputs/webly%A%a.out
#SBATCH --error=errors/webly%A%a.err

echo Hello running $SLURM_ARRAY_TASK_ID
source $PROJECT/menna/.weblyvenv/bin/activate
python main.py -train 1 -test_multi_run 1 -model_type iter_nwe_coatt -fold 0 -embed_type word2vec -data_dir $PROJECT/menna/VOCdevkit/VOC2012/ -dataset_name pascal -n_shots 1 -bs 4 -ckpt ckpt_iter_nwe_coatt
