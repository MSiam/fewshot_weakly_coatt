import random
import os
import torchvision
import torch
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as F_tensor
import numpy as np
from torch.utils.data import DataLoader
import time
import glob

class Dataset(object):


    def __init__(self, data_dir, fold, input_size=[321, 321], normalize_mean=[0, 0, 0],
                 normalize_std=[1, 1, 1], seed=None, split='val'):

        self.data_dir = data_dir
        self.input_size = input_size

        self.rand = random.Random()
        self.split = split
        #random sample 1000 pairs
        self.seed = seed
        if seed is not None:
            self.rand.seed(seed)
        self.chosen_data_list_1 = self.get_new_exist_class_dict(fold=fold)
        chosen_data_list_2 = self.chosen_data_list_1[:]
        chosen_data_list_3 = self.chosen_data_list_1[:]
        self.rand.shuffle(chosen_data_list_2)
        self.rand.shuffle(chosen_data_list_3)
        self.chosen_data_list=self.chosen_data_list_1+chosen_data_list_2+chosen_data_list_3
        self.chosen_data_list=self.chosen_data_list[:1000]

        self.binary_pair_list = self.get_binary_pair_list()#a dict of each class, which contains all imgs that include this class
        self.history_mask_list = [None] * 1000
        self.query_class_support_list=[None] * 1000
        for index in range (1000):
            query_name=self.chosen_data_list[index][0]
            sample_class=self.chosen_data_list[index][1]
            support_img_list = self.binary_pair_list[sample_class]  # all img that contain the sample_class
            while True:  # random sample a support data
                support_name = support_img_list[self.rand.randint(0, len(support_img_list) - 1)]
                if support_name != query_name:
                    break
            self.query_class_support_list[index]=[query_name,sample_class,support_name]

        self.initiaize_transformation(normalize_mean, normalize_std, input_size)
        pass

    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []

        if self.split == 'train':
            fold_list=[0,1,2,3]
            fold_list.remove(fold)
            for fold in fold_list:
                f = open(os.path.join(self.data_dir, 'Binary_map_aug', 'val', 'split%1d_val.txt'%fold))
                while True:
                    item = f.readline()
                    if item == '':
                        break
                    img_name = item[:11]
                    cat = int(item[13:15])
                    new_exist_class_list.append([img_name, cat])
            self.split = 'val' # Validates on validation images but using training classes
        else:
            f = open(os.path.join(self.data_dir, 'Binary_map_aug', self.split, \
                     'split%1d_'%(fold)+ self.split + '.txt'))
            while True:
                item = f.readline()
                if item == '':
                    break
                img_name = item[:11]
                cat = int(item[13:15])
                new_exist_class_list.append([img_name, cat])
        return new_exist_class_list

    def initiaize_transformation(self, normalize_mean, normalize_std, input_size):
        self.ToTensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(normalize_mean, normalize_std)

    def get_binary_pair_list(self):  # a list store all img name that contain that class
        binary_pair_list = {}
        for Class in range(1, 21):
            binary_pair_list[Class] = self.read_txt(
                os.path.join(self.data_dir, 'Binary_map_aug', self.split, '%d.txt' % Class))
        return binary_pair_list

    def read_txt(self, dir):
        f = open(dir)
        out_list = []
        line = f.readline()
        while line:
            out_list.append(line.split()[0])
            line = f.readline()
        return out_list

    def __getitem__(self, index):
        # give an query index,sample a target class first
        query_name = self.query_class_support_list[index][0]
        sample_class = self.query_class_support_list[index][1]  # random sample a class in this img
        support_name=self.query_class_support_list[index][2]


        input_size = self.input_size[0]
        # random scale and crop for support
        scaled_size = int(self.rand.uniform(1,1.5)*input_size)
        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)

        flip_flag = self.rand.random()
        margin_h = self.rand.randint(0, scaled_size - input_size)
        margin_w = self.rand.randint(0, scaled_size - input_size)

        support_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,
                              Image.open(
                                  os.path.join(self.data_dir, 'JPEGImages', support_name + '.jpg'))))))

        support_original = np.array(
                                scale_transform_rgb(
                                    Image.open(os.path.join(self.data_dir, 'JPEGImages', support_name + '.jpg'))))
        support_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'Binary_map_aug', self.split, str(sample_class),
                                           support_name + '.png')))))
        support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_mask = support_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_original = support_original[margin_h:margin_h + input_size, margin_w:margin_w + input_size, :]


        # random scale and crop for query
        scaled_size = self.input_size[0]

        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = 0#random.random()

        query_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,
                              Image.open(
                                  os.path.join(self.data_dir, 'JPEGImages', query_name + '.jpg'))))))

        qry_original = np.array(
                            scale_transform_rgb(
                                Image.open(os.path.join(self.data_dir, 'JPEGImages', query_name + '.jpg'))))

        query_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'Binary_map_aug', self.split, str(sample_class),
                                           query_name + '.png')))))
        margin_h = self.rand.randint(0, scaled_size - input_size)
        margin_w = self.rand.randint(0, scaled_size - input_size)

        query_rgb = query_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_mask = query_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        if self.history_mask_list[index] is None:

            history_mask=torch.zeros(2,41,41).fill_(0.0)

        else:

            history_mask=self.history_mask_list[index]

        return query_rgb, query_mask, support_rgb, support_mask,history_mask, \
                    support_original, qry_original, sample_class,index

    def flip(self, flag, img):
        if flag > 0.5:
            return F.hflip(img)
        else:
            return img

    def __len__(self):
        return 1000

class OSLSMSetupDataset(Dataset):
    def __init__(self, data_dir, fold, input_size=[500, 500],
                 normalize_mean=[0, 0, 0], normalize_std=[1, 1, 1],
                 seed=None):

        super(OSLSMSetupDataset, self).__init__(data_dir, fold, input_size,
                                                normalize_mean, normalize_std,
                                                seed)
        k_shot = 1
        self.query_class_support_list = self.parse_file(
            os.path.join(self.data_dir, 'data_files', 'imgs_paths_%d_%d.txt'%(fold, k_shot)),
                                           k_shot)

    def parse_file(self, pth_txt, k_shot):
        files = []
        pair = []
        support = []
        f = open(pth_txt, 'r')

        count = 0
        for line in f:
            if count == (k_shot+1)*2:
                pair.insert(1, int(line.split(' ')[-1].strip()))
                files.append(pair)
                count = -1
            elif count < k_shot:
                support.append(line.strip().split('/')[2].split('.')[0])
            elif count < k_shot+1:
                pair = [line.strip().split('/')[2].split('.')[0],
                        support[0]] #[0] is temporary considering 1shot only
                support = []
            count += 1
        return files

    def __getitem__(self, index):
        # give an query index,sample a target class first
        query_name = self.query_class_support_list[index][0]
        sample_class = self.query_class_support_list[index][1]  # random sample a class in this img
        support_name=self.query_class_support_list[index][2]

        input_size = self.input_size[0]
        # random scale and crop for support
        scaled_size = int(self.rand.uniform(1,1.5)*input_size)
        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)

        flip_flag = 0
        margin_h = 0
        margin_w = 0

        support_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,
                              Image.open(
                                  os.path.join(self.data_dir, 'JPEGImages', support_name + '.jpg'))))))

        support_original = np.array(
                                scale_transform_rgb(
                                    Image.open(os.path.join(self.data_dir, 'JPEGImages', support_name + '.jpg'))))
        support_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'Binary_map_aug', self.split, str(sample_class),
                                           support_name + '.png')))))
        support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_mask = support_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_original = support_original[margin_h:margin_h + input_size, margin_w:margin_w + input_size, :]


        # random scale and crop for query
        scaled_size = self.input_size[0]

        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = 0

        query_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,
                              Image.open(
                                  os.path.join(self.data_dir, 'JPEGImages', query_name + '.jpg'))))))

        qry_original = np.array(
                            scale_transform_rgb(
                                Image.open(os.path.join(self.data_dir, 'JPEGImages', query_name + '.jpg'))))

        query_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'Binary_map_aug', self.split, str(sample_class),
                                           query_name + '.png')))))
        margin_h = 0
        margin_w = 0

        query_rgb = query_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_mask = query_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        if self.history_mask_list[index] is None:

            history_mask=torch.zeros(2,41,41).fill_(0.0)

        else:

            history_mask=self.history_mask_list[index]

        return query_rgb, query_mask, support_rgb, support_mask,history_mask, \
                    support_original, qry_original, sample_class,index

class WebSetupDataset(OSLSMSetupDataset):
    def __init__(self, data_dir, fold, input_size=[500, 500],
                 normalize_mean=[0, 0, 0], normalize_std=[1, 1, 1],
                 seed=None):

        super(WebSetupDataset, self).__init__(data_dir, fold, input_size,
                                              normalize_mean, normalize_std,
                                              seed)
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']

        self.modify_sprt_to_web('google_', fold)

    def modify_sprt_to_web(self, prefix_pth, fold):
        for it, (qry, cls, sprt) in enumerate(self.query_class_support_list):
            files = sorted(glob.glob(self.data_dir+'/'+prefix_pth+str(fold)+'/'+\
                                self.classes[cls-1]+'*'))
            self.query_class_support_list[it].append(files[0])

    def __getitem__(self, index):
        query_rgb, query_mask, _, support_mask,\
            history_mask, _, qry_original, \
            sample_class,index = super(WebSetupDataset, self).__getitem__(index)

        scaled_size = self.input_size[0]
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                            interpolation=Image.BILINEAR)
        support_name = self.query_class_support_list[index][-1]

        support_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                        Image.open(support_name))))

        support_original = np.array(
                                scale_transform_rgb(
                                    Image.open(support_name )))

        return query_rgb, query_mask, support_rgb, support_mask,history_mask, \
                    support_original, qry_original, sample_class,index
