"""
Load pascal VOC dataset
"""

import os

import numpy as np
from PIL import Image
import torch

from .common import BaseDataset
from pdb import set_trace as bkpt

class VOC(BaseDataset):
    """
    Base Class for VOC Dataset

    Args:
        base_dir:
            VOC dataset directory
        split:
            which split to use
            choose from ('train', 'val', 'trainval', 'trainaug')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
    """
    def __init__(self, base_dir, split, transforms=None, to_tensor=None):
        super().__init__(base_dir)
        self.split = split
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._label_dir = os.path.join(self._base_dir, 'SegmentationClassAug')
        self._inst_dir = os.path.join(self._base_dir, 'SegmentationObjectAug')
        self._scribble_dir = os.path.join(self._base_dir, 'ScribbleAugAuto')
        self._id_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')
        self.transforms = transforms
        self.to_tensor = to_tensor
#       self.labels = labels
        #self.new_exist_class_list = self.get_new_exist_class_dict(fold=fold)

        with open(os.path.join(self._id_dir, f'{self.split}.txt'), 'r') as f:
            self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.ids)

    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []

        fold_list=[0,1,2,3]
        fold_list.remove(fold)
        for fold in fold_list:

            f = open(os.path.join(self._base_dir, 'Binary_map_aug', 'train', 'split%1d_train.txt'%fold))
            while True:
                item = f.readline()
                if item == '':
                    break
                img_name = item[:11]
                cat = int(item[13:15])
                new_exist_class_list.append([img_name, cat])
                return new_exist_class_list

    def __getitem__(self, idx):
        # Fetch data
        id_ = self.ids[idx]
        
        #id_ = self.new_exist_class_list[index][0]
        #sample_class = self.new_exist_class_list[index][1]        
        
        image = Image.open(os.path.join(self._image_dir, f'{id_}.jpg'))
        semantic_mask = Image.open(os.path.join(self._label_dir, f'{id_}.png'))
        instance_mask = Image.open(os.path.join(self._inst_dir, f'{id_}.png'))
        scribble_mask = Image.open(os.path.join(self._scribble_dir, f'{id_}.png'))
        sample = {'image': image,
                  'label': semantic_mask,
                  'inst': instance_mask,
                  'scribble': scribble_mask}

        # Image-level transformation
        if self.transforms is not None:
            sample = self.transforms(sample)
        # Save the original image (without normalization)
        image_t = torch.from_numpy(np.array(sample['image']).transpose(2, 0, 1))
        # Transform to tensor
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)
        bkpt()
        sample['id'] = id_
        sample['image_t'] = image_t
#        sample['sample_class'] = sample_class
        # Add auxiliary attributes
        for key_prefix in self.aux_attrib:
            # Process the data sample, create new attributes and save them in a dictionary
            aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
            for key_suffix in aux_attrib_val:
                # one function may create multiple attributes, so we need suffix to distinguish them
                sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]

        return sample
