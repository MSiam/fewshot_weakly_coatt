
DATA_DIR=/mnt/datasets/public/research/pascal/

# download datasets
python pascal_voc_download.py --dataset-dir $DATA_DIR
python sbd_download.py --dataset-dir $DATA_DIR

# overrides train.txt to use train_aug for SBD data ensuring no overlap with val data
cp train_aug.txt "$DATA_DIR"/ImageSets/Segmentation/train.txt

# use binary_aug data from CANet
unzip Binary_map_aug.zip
cp -r Binary_map_aug/ $DATA_DIR
