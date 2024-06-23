from torch.utils.data import Dataset as dts

import jsonlines
import cv2
import os

import numpy as np

class Dataset(dts):
    def __init__(self, 
                 data_path: str,
                 names: list):
        self.data_dirname = os.path.dirname(data_path)
        self.label2idx = {label:idx for idx, label in enumerate(names)}
        
        with jsonlines.open(data_path) as reader:
            self.samples = [obj for obj in reader]

    def __len__(self,):
        return len(self.samples)
    
    def __getitem__(self, idx):
        current_sample = self.samples[idx]
        img_p = os.path.join(self.data_dirname, current_sample["image"])
        label = current_sample["label"]

        tensor = self._load_preprocess(img_p)
        label_idx = self.label2idx[label]

        return tensor, label_idx

    def _load_preprocess(self, img_p: str):
        bgr_image = cv2.imread(img_p)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (224,224))

        tensor = np.divide(rgb_image, 255)
        tensor = np.transpose(tensor, (2,0,1))

        return tensor
        
        



