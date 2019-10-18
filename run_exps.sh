for fold in {3..3}
do
    echo 'running fold '$fold
    python train.py -fold $fold -ckpt checkpoint_coatt_nwe -model_type nwe_coatt
done
