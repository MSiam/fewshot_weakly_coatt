
DATA_DIR=/mnt/datasets/public/research/pascal/VOCdevkit/VOC2012/

# download datasets
python pascal_voc_download.py --dataset-dir $DATA_DIR
python sbd_download.py --dataset-dir $DATA_DIR

# overrides train.txt to use train_aug for SBD data ensuring no overlap with val data
cp train_aug.txt "$DATA_DIR"/ImageSets/Segmentation/train.txt

# use binary_aug data from CANet
unzip Binary_map_aug.zip
cp -r Binary_map_aug/ $DATA_DIR

# copy pretrained text embeddings to the data directory
cp embeddings.npy $DATA_DIR
cp concatenated_embed.npy $DATA_DIR
cp fsttxt.npy $DATA_DIR

# copy fold data
cp -r data_files/ $DATA_DIR
