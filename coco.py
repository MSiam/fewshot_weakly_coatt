"""
Load COCO dataset
"""

from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import torch

from common_coco import BaseDataset, PairedDataset
from torchvision.transforms import Compose
import torchvision.transforms.functional as tr_F
import random
import matplotlib.pyplot as plt
from torch.utils import data

class COCOSeg(BaseDataset):
    """
    Modified Class for COCO Dataset

    Args:
        base_dir:
            COCO dataset directory
        split:
            which split to use (default is 2014 version)
            choose from ('train', 'val')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
    """
    def __init__(self, base_dir, split, transforms=None, to_tensor=None):
        super().__init__(base_dir)
        self.split = split + '2014'
        annFile = f'{base_dir}/annotations/instances_{self.split}.json'
        print('Loading file ', annFile)
        self.coco = COCO(annFile)

        self.ids = self.coco.getImgIds()
        self.transforms = transforms
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Fetch meta data
        id_ = self.ids[idx]
        img_meta = self.coco.loadImgs(id_)[0]
        annIds = self.coco.getAnnIds(imgIds=img_meta['id'])

        # Open Image
        image = Image.open(f"{self._base_dir}/{self.split}/{img_meta['file_name']}")
        if image.mode == 'L':
            image = image.convert('RGB')

        # Process masks
        anns = self.coco.loadAnns(annIds)
        semantic_masks = {}
        for ann in anns:
            catId = ann['category_id']
            mask = self.coco.annToMask(ann)
            if catId in semantic_masks:
                semantic_masks[catId][mask == 1] = catId
            else:
                semantic_mask = np.zeros((img_meta['height'], img_meta['width']), dtype='uint8')
                semantic_mask[mask == 1] = catId
                semantic_masks[catId] = semantic_mask
        semantic_masks = {catId: Image.fromarray(semantic_mask)
                          for catId, semantic_mask in semantic_masks.items()}

        # Filter out small masks
        temp_semantic_masks = semantic_masks.copy()
        for catId, mask in semantic_masks.items():
            mask_array = np.array(mask).copy()
            mask_array[mask_array==catId] = 1
            if np.sum(mask_array[mask_array==1]) < 10: # remove objects less than 10 pixels!
                del temp_semantic_masks[catId]
        semantic_masks = temp_semantic_masks

        sample = {'image': image,
                  'label': semantic_masks}

        # Image-level transformation
        if self.transforms is not None:
            sample = self.transforms(sample)
        # Save the original image (without mean subtraction/normalization)
        image_t = torch.from_numpy(np.array(sample['image']).transpose(2, 0, 1))

        # Transform to tensor
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = id_
        sample['image_t'] = image_t

        # Add auxiliary attributes
        for key_prefix in self.aux_attrib:
            # Process the data sample, create new attributes and save them in a dictionary
            aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
            for key_suffix in aux_attrib_val:
                # one function may create multiple attributes, so we need suffix to distinguish them
                sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]

        return sample

class ToTensorNormalize(object):
    """
    Convert images/masks to torch.Tensor
    Scale images' pixel values to [0-1] and normalize with predefined statistics
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = tr_F.to_tensor(img)
        img = tr_F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if isinstance(label, dict):
            label = {catId: torch.Tensor(np.array(x)).long()
                     for catId, x in label.items()}
        else:
            label = torch.Tensor(np.array(label)).long()

        sample['image'] = img
        sample['label'] = label
        return sample

class RandomMirror(object):
    """
    Randomly filp the images/masks horizontally
    """
    def __init__(self, seed=1337):
        self.rand_gen = random.Random(seed)

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        if self.rand_gen.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if isinstance(label, dict):
                label = {catId: x.transpose(Image.FLIP_LEFT_RIGHT)
                         for catId, x in label.items()}
            else:
                label = label.transpose(Image.FLIP_LEFT_RIGHT)

        sample['image'] = img
        sample['label'] = label
        return sample

class Resize(object):
    """
    Resize images/masks to given size
    Args:
        size: output size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = tr_F.resize(img, self.size)
        if isinstance(label, dict):
            label = {catId: tr_F.resize(x, self.size, interpolation=Image.NEAREST)
                     for catId, x in label.items()}
        else:
            label = tr_F.resize(label, self.size, interpolation=Image.NEAREST)

        sample['image'] = img
        sample['label'] = label
        return sample

def attrib_basic(_sample, class_id):
    """
    Add basic attribute
    Args:
        _sample: data sample
        class_id: class label asscociated with the data
            (sometimes indicting from which subset the data are drawn)
    """
    return {'class_id': class_id}

def getMask(label, class_id, class_ids):
    """
    Generate FG/BG mask from the segmentation mask
    Args:
        label:
            semantic mask
        class_id:
            semantic class of interest
        class_ids:
            all class id in this episode
    """
    # Dense Mask
    fg_mask = torch.where(label == class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    bg_mask = torch.where(label != class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    for class_id in class_ids:
        bg_mask[label == class_id] = 0

    return {'fg_mask': fg_mask,
            'bg_mask': bg_mask}

def fewShot(paired_sample, n_ways, n_shots, cnt_query, coco=False):
    """
    Postprocess paired sample for fewshot settings
    Args:
        paired_sample:
            data sample from a PairedDataset
        n_ways:
            n-way few-shot learning
        n_shots:
            n-shot few-shot learning
        cnt_query:
            number of query images for each class in the support set
        coco:
            MS COCO dataset
    """
    ###### Compose the support and query image list ######
    cumsum_idx = np.cumsum([0,] + [n_shots + x for x in cnt_query])

    # support class ids
    class_ids = [paired_sample[cumsum_idx[i]]['basic_class_id'] for i in range(n_ways)]
    # support images
    support_images = [[paired_sample[cumsum_idx[i] + j]['image'] for j in range(n_shots)]
                      for i in range(n_ways)]
    support_images_t = [[paired_sample[cumsum_idx[i] + j]['image_t'] for j in range(n_shots)]
                        for i in range(n_ways)] # original support images

    # support image labels
    support_labels = [[paired_sample[cumsum_idx[i] + j]['label'][class_ids[i]]
                       for j in range(n_shots)] for i in range(n_ways)]

    # query images, masks and class indices
    query_images = [paired_sample[cumsum_idx[i+1] - j - 1]['image'] for i in range(n_ways)
                    for j in range(cnt_query[i])]
    query_images_t = [paired_sample[cumsum_idx[i+1] - j - 1]['image_t'] for i in range(n_ways)
                      for j in range(cnt_query[i])]
    query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'][class_ids[i]]
                    for i in range(n_ways) for j in range(cnt_query[i])]
    query_cls_idx = [sorted([0,] + [class_ids.index(x) + 1
                                    for x in set(np.unique(query_label)) & set(class_ids)])
                     for query_label in query_labels] # indices for labels not original catids but rather enumeration

    ###### Generate support image masks ######
    support_mask = [[getMask(support_labels[way][shot], class_ids[way], class_ids)
                     for shot in range(n_shots)] for way in range(n_ways)]

    ###### Generate query label (class indices in one episode, i.e. the ground truth)######
    query_labels_tmp = [torch.zeros_like(x) for x in query_labels]
    for i, query_label_tmp in enumerate(query_labels_tmp):
        query_label_tmp[query_labels[i] == 255] = 255
        for j in range(n_ways):
            query_label_tmp[query_labels[i] == class_ids[j]] = j + 1

    ###### Generate query mask for each semantic class (including BG) ######
    # BG class
    query_masks = [[torch.where(query_label == 0,
                                torch.ones_like(query_label),
                                torch.zeros_like(query_label))[None, ...],]
                   for query_label in query_labels]
    # Other classes in query image
    for i, query_label in enumerate(query_labels):
        for idx in query_cls_idx[i][1:]:
            mask = torch.where(query_label == class_ids[idx - 1],
                               torch.ones_like(query_label),
                               torch.zeros_like(query_label))[None, ...]
            query_masks[i].append(mask)

    return {'class_ids': class_ids,
            'support_images_t': support_images_t,
            'support_images': support_images,
            'support_mask': support_mask,
            'query_images_t': query_images_t,
            'query_images': query_images,
            'query_labels': query_labels_tmp,
            'query_masks': query_masks,
            'query_cls_idx': query_cls_idx,
           }

def create_coco_fewshot(base_dir, split, input_size,
                        n_ways, n_shots, max_iters, fold,
                        prob, seed=1337, n_queries=1):

    transforms = Compose([Resize(size=input_size),
                          RandomMirror(seed=seed)])

    to_tensor = ToTensorNormalize()

    CLASS_LABELS = { 'train': {
            0: set(range(1, 81)) - set(range(1, 21)),
            1: set(range(1, 81)) - set(range(21, 41)),
            2: set(range(1, 81)) - set(range(41, 61)),
            3: set(range(1, 81)) - set(range(61, 81)),
        },
        'val': {
            0: set(range(1, 21)),
            1: set(range(21, 41)),
            2: set(range(41, 61)),
            3: set(range(61, 81)),
        }
    }

    if split == 'trainval':
        labels = CLASS_LABELS['train'][fold]
        split = 'val'
    elif split == 'test':
        labels = CLASS_LABELS['val'][fold]
        split = 'val'
    else: #'train'
        labels = CLASS_LABELS[split][fold]

    cocoseg = COCOSeg(base_dir, split, transforms, to_tensor)
    cocoseg.add_attrib('basic', attrib_basic, {})

    # Load image ids for each class
    cat_ids = cocoseg.coco.getCatIds()
    sub_ids = [cocoseg.coco.getImgIds(catIds=cat_ids[i - 1]) for i in labels]

    # Create sub-datasets and add class_id attribute
    subsets = cocoseg.subsets(sub_ids, [{'basic': {'class_id': cat_ids[i - 1]}} for i in labels])

    # Choose the classes of queries
    rand_gen = random.Random(seed)
    cnt_query = np.bincount(rand_gen.choices(population=range(n_ways), k=n_queries),
                            minlength=n_ways)

    # Set the number of images for each class
    n_elements = [n_shots + x for x in cnt_query]

    # Create paired dataset
    paired_data = PairedDataset(subsets, n_elements=n_elements, max_iters=max_iters, same=False,
                                pair_based_transforms=[
                                    (fewShot, {'n_ways': n_ways, 'n_shots': n_shots,
                                               'cnt_query': cnt_query, 'coco': True})],
                                seed=seed, prob=prob)
    return paired_data, cat_ids

if __name__ ==  "__main__":
    dataset, cat_ids = create_coco_fewshot('/home/msiam/Dataset/COCO/', 'trainval', input_size=(321, 321),
                                          n_ways=1, n_shots=1, max_iters=30000, fold=1,
                                          prob=0.6, seed=1337)
    dataloader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0,
                                drop_last=False)


    for iter, sample in enumerate(dataloader):
        print('Iteration ', iter)
        qry_img, qry_mask, sprt_img, sprt_mask, history, sprt_original, qry_original, cls, idx = sample
       # plt.figure(1); plt.imshow(np.transpose(qry_original, (1,2,0)));
       # plt.figure(2); plt.imshow(np.transpose(sprt_original, (1,2,0)));
       # plt.figure(3); plt.imshow(qry_mask[0]);
       # plt.figure(4); plt.imshow(sprt_mask[0]);plt.show()
        print('Class # ', cls)
       # import pdb; pdb.set_trace()
