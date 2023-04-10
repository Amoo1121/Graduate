import torch
from torch.utils import data
import os
import cv2
import numpy as np

    
class TrainData(data.Dataset):
    def __init__(self, cfg):
        path = cfg.train_data_path
        imgs = os.listdir(path)
        self.imgs = [os.path.join(path, img) for img in imgs]

    def __getitem__(self, key):
        img_path = self.imgs[key]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256)).astype('float32')
        img = (img / 127.5) - 1.0 
        img= np.transpose(img, [2, 0, 1])
        img = torch.from_numpy(img)
        return img

    def __len__(self):
        return len(self.imgs)
