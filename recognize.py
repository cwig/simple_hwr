import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

import crnn

import character_set
import os
import sys
import cv2
import numpy as np

import random
import string_utils

def main():
    config_path = sys.argv[1]
    image_path = sys.argv[2]

    with open(config_path) as f:
        config = json.load(f)

    idx_to_char, char_to_idx = character_set.load_char_set(config['character_set_path'])

    hw = crnn.create_model({
        'cnn_out_size': config['network']['cnn_out_size'],
        'num_of_channels': 3,
        'num_of_outputs': len(idx_to_char)+1
    })

    hw.load_state_dict(torch.load(config['model_save_path']))
    if torch.cuda.is_available():
        hw.cuda()
        dtype = torch.cuda.FloatTensor
        print("Using GPU")
    else:
        dtype = torch.FloatTensor
        print("No GPU detected")

    hw.eval()

    img = cv2.imread(image_path)
    if img.shape[0] != config['network']['input_height']:
        percent = float(config['network']['input_height']) / img.shape[0]
        img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)


    img = torch.from_numpy(img.transpose(2,0,1).astype(np.float32)/128 - 1)
    img = Variable(img[None,...].type(dtype), requires_grad=False, volatile=True)

    preds = hw(img)

    output_batch = preds.permute(1,0,2)
    out = output_batch.data.cpu().numpy()

    pred, pred_raw = string_utils.naive_decode(out[0])
    pred_str = string_utils.label2str(pred, idx_to_char, False)
    pred_raw_str = string_utils.label2str(pred_raw, idx_to_char, True)
    print pred_raw_str
    print pred_str

if __name__ == "__main__":
    main()
