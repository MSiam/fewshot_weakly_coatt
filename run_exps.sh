#python main.py -fold 0 -train 0 -ckpt checkpoint_coatt_nwe -film 0 -model_type nwe_coatt -data_dir ~/Dataset/VOCdevkit/VOC2012/ -dataset_name pascal -test_multi_run 1 -bs 2 -n_shots 1
python main.py -fold 0 -train 0 -ckpt checkpoint_coatt_nwe -film 0 -model_type nwe_coatt -data_dir ~/Dataset/VOCdevkit/VOC2012/ -dataset_name pascal -test_multi_run 1 -bs 2 -n_shots 5


#for fold in {3..3}
#do
#    echo 'running fold '$fold
#    python train.py -fold $fold -ckpt checkpoint_coatt_nwe -model_type nwe_coatt
#done
