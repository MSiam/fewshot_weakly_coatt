DATA_DIR=~/Data/VOCdevkit/VOC2012/
CKPT_DIR=iter_nwe_coatt/

for fold in {0..3}
do
    python main.py -train 1 -test_multi_run 1 -model_type iter_nwe_coatt -fold 0 -embed_type word2vec -data_dir $DATA_DIR -dataset_name pascal -n_shots 1 -bs 4 -ckpt $CKPT_DIR -reproducability 1 -num_epoch 50 -lr 0.001 -gamma_steplr 0.1
done

