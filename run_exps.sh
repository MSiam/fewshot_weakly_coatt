#python main.py -fold 0 -train 0 -ckpt checkpoint_coatt_nwe -film 0 -model_type nwe_coatt -data_dir ~/Dataset/VOCdevkit/VOC2012/ -dataset_name pascal -test_multi_run 1 -bs 2 -n_shots 1
#python main.py -fold 0 -train 0 -ckpt checkpoint_coatt_nwe -film 0 -model_type nwe_coatt -data_dir ~/Dataset/VOCdevkit/VOC2012/ -dataset_name pascal -test_multi_run 1 -bs 2 -n_shots 5


#for fold in {3..3}
#do
#    echo 'running fold '$fold
#    python train.py -fold $fold -ckpt checkpoint_coatt_nwe -model_type nwe_coatt
#done

#python main.py -train 0 -test_multi_run 1 -model_type iter_nwe_coatt -fold 0 -embed_type word2vec -data_dir ~/Dataset/VOCdevkit/VOC2012/ -dataset_name pascal -n_shots 5 -bs 1 -ckpt checkpoints_trainval/model_type\=iter_nwe_coatt\,fold\=0\,seed\=1337\,lr\=0.001\,split\=trainval\,embed_type\=word2vec\,dataset_name\=pascal/testing/

#batch_sizes=(8)
#n_shots=(1)
##batch_sizes=(8, 1)
##n_shots=(1, 5)
#for i in {0..1}
#do
#    for fold in {0..3}
#    do
#        echo 'N_SHOTS =' ${n_shots[i]} ' FOLD = ' $fold ' BS = ' ${batch_sizes[i]}
#        python main.py -train 0 -test_multi_run 1 -model_type iter_nwe_coatt -fold $fold -embed_type word2vec -data_dir ~/Dataset/VOCdevkit/VOC2012/ -dataset_name pascal -n_shots ${n_shots[i]} -bs ${batch_sizes[i]} -ckpt checkpoints_trainval/model_type\=iter_nwe_coatt\,fold\=$fold\,seed\=1337\,lr\=0.001\,split\=trainval\,embed_type\=word2vec\,dataset_name\=pascal/testing/ -use_web 1
#    done
#done

python main.py -train 1 -test_multi_run 1 -model_type iter_nwe_coatt -fold 0 -embed_type word2vec -data_dir ~/Dataset/VOCdevkit/VOC2012/ -dataset_name pascal -n_shots 1 -bs 4 -ckpt stability_exps/test1/ -reproducability 1 -num_epoch 20
python main.py -train 1 -test_multi_run 1 -model_type iter_nwe_coatt -fold 0 -embed_type word2vec -data_dir ~/Dataset/VOCdevkit/VOC2012/ -dataset_name pascal -n_shots 1 -bs 4 -ckpt stability_exps/test2/ -reproducability 1 -num_epoch 20
python main.py -train 1 -test_multi_run 1 -model_type iter_nwe_coatt -fold 0 -embed_type word2vec -data_dir ~/Dataset/VOCdevkit/VOC2012/ -dataset_name pascal -n_shots 1 -bs 4 -ckpt stability_exps/test3/ -reproducability 1 -num_epoch 20
