#!/bin/bash
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --job-name=fewshot_webly
#SBATCH --gres=gpu:2
#SBATCH --mem=5000M
#SBATCH --time=0-20:00 #d-hh:mm
#SBATCH --output=outputs/webly%A%a.out
#SBATCH --error=errors/webly%A%a.err

#folds=(0 1 2 3)
n_shots=1
#models=('iter_nwe_coatt' 'iter_nwe_coatt')
#experiments=()
#
#for fold in {0..3}
#do
#        for i in {0..1}
#        do
#                experiments+=($fold ${models[i]})
#        done
#done

experiments=( 0 'iter_nwe_coatt' 1 'iter_nwe_coatt' 2 'iter_nwe_coatt' 3 'iter_nwe_coatt')
#echo Hello running $SLURM_ARRAY_TASK_ID fold ${experiments[$((($SLURM_ARRAY_TASK_ID-1)*2))]} model type ${experiments[$((($SLURM_ARRAY_TASK_ID-1)*2+1))]}

source /home/menna/projects/def-jag/menna/.weblyvenv/bin/activate
python main.py -fold ${experiments[$((($SLURM_ARRAY_TASK_ID-1)*2))]} -train 1 -test_multi_run 1 -model_type ${experiments[$((($SLURM_ARRAY_TASK_ID-1)*2+1))]} -embed_type word2vec -data_dir /home/menna/projects/def-jag/menna/VOCdevkit/VOC2012/ -dataset_name pascal -n_shots $n_shots -bs 4 -ckpt ${experiments[$((($SLURM_ARRAY_TASK_ID-1)*2+1))]} -niterations 100000
