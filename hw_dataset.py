import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import cv2
import numpy as np

import random
import string_utils

import grid_distortion

PADDING_CONSTANT = 0

def collate(batch):
    batch = [b for b in batch if b is not None]
    #These all should be the same size or error
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    dim0 = batch[0]['line_img'].shape[0]
    dim1 = max([b['line_img'].shape[1] for b in batch])
    dim2 = batch[0]['line_img'].shape[2]

    all_labels = []
    label_lengths = []

    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
    for i in xrange(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i,:,:b_img.shape[1],:] = b_img

        l = batch[i]['gt_label']
        all_labels.append(l)
        label_lengths.append(len(l))

    all_labels = np.concatenate(all_labels)
    label_lengths = np.array(label_lengths)

    line_imgs = input_batch.transpose([0,3,1,2])
    line_imgs = torch.from_numpy(line_imgs)
    labels = torch.from_numpy(all_labels.astype(np.int32))
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32))

    return {
        "line_imgs": line_imgs,
        "labels": labels,
        "label_lengths": label_lengths,
        "gt": [b['gt'] for b in batch]
    }

class HwDataset(Dataset):
    def __init__(self, json_path, char_to_idx, img_height=32, root_path=".", augmentation=False):
        with open(json_path) as f:
            data = json.load(f)

        self.root_path = root_path
        self.img_height = img_height
        self.char_to_idx = char_to_idx
        self.data = data
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img = cv2.imread(os.path.join(self.root_path, item['image_path']))

        if img is None:
            print("Warning: image is None:", os.path.join(self.root_path, item['image_path']))
            return None


        percent = float(self.img_height) / img.shape[0]
        img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)

        if self.augmentation:
            img = grid_distortion.warp_image(img) 

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0

        gt = item['gt']
        gt_label = string_utils.str2label(gt, self.char_to_idx)

        return {
            "line_img": img,
            "gt_label": gt_label,
            "gt": gt
        }
