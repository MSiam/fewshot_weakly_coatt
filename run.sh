#!/bin/bash
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --job-name=fewshot_webly
#SBATCH --gres=gpu:1
#SBATCH --mem=5000M
#SBATCH --time=2-10:00 #d-hh:mm
#SBATCH --output=outputs/webly%A%a.out
#SBATCH --error=errors/webly%A%a.err

folds=(0 1)
n_shots=1
models=('iter_nwe_coatt' 'iter_nwe_coatt')
experiments=()

for fold in {0..1}
do
        for i in {0..1}
        do
                experiments+=($fold ${models[i]})
        done
done

echo Hello running $SLURM_ARRAY_TASK_ID fold ${experiments[$((($SLURM_ARRAY_TASK_ID-1)*2))]} model type ${experiments[$((($SLURM_ARRAY_TASK_ID-1)*2+1))]}

source $PROJECT/menna/.weblyvenv/bin/activate
python main.py -fold ${experiments[$((($SLURM_ARRAY_TASK_ID-1)*2))]} -train 1 -test_multi_run 1 -model_type ${experiments[$((($SLURM_ARRAY_TASK_ID-1)*2+1))]} -embed_type word2vec -data_dir $PROJECT/menna/VOCdevkit/VOC2012/ -dataset_name pascal -n_shots $n_shots -bs 4 -ckpt ${experiments[$((($SLURM_ARRAY_TASK_ID-1)*2+1))]}_$SLURM_ARRAY_TASK_ID -reproducability 1
