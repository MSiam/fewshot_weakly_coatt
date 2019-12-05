
DATA_DIR=$PROJECT/menna/
DATA_DIR_COCO=/mnt/datasets/public/research/COCO

# download datasets
python pascal_voc_download.py --dataset-dir $DATA_DIR
python sbd_download.py --dataset-dir $DATA_DIR
python coco_download.py --dataset-dir $DATA_DIR_COCO

# overrides train.txt to use train_aug for SBD data ensuring no overlap with val data
cp train_aug.txt "$DATA_DIR"/ImageSets/Segmentation/train.txt

# use binary_aug data from CANet
unzip Binary_map_aug.zip
cp -r Binary_map_aug/ $DATA_DIR

# copy pretrained text embeddings to the data directory
cp embeddings_*pascal.npy $DATA_DIR
cp embeddings_*coco.npy $DATA_DIR_COCO

# copy fold data
cp -r data_files/ $DATA_DIR

# copy COCO classes
cp coco_classes.txt $DATA_DIR_COCO
