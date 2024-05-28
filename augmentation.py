import csv
import os
import random
import torchvision

import imgaug.augmenters as ia
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class augmentation(object):
    def __init__(self, path, views, labels, base_dir=None, transform=True, normalize=True, Test= False) -> None:
        self.data = []
        self.transform = transform
        self.normalize = normalize
        self.views = views
        self.labels = labels
        self.base_dir = base_dir
        self.Test=Test
        self.initialize_data(path)
        import imgaug.augmenters as iaa
        aug = iaa.Sequential([
            ia.pillike.FilterEdgeEnhance(),
            ia.pillike.FilterFindEdges(),
            ia.pillike.FilterContour(),
            ia.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            ia.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5)
            ia.Invert(0.5)
            ia.Solarize(0.5, threshold=(32, 128))
            ia.AdditivePoissonNoise(40)
        ])

    def __getitem__(self, i) -> dict:
        item = self.data[i]
        output = {}
        for view in self.views:
            img = item[view]
            if self.base_dir is not None:
                img = os.path.join(self.base_dir, img)
            output[view] = self._process_view(self._load_img(img))
        label = np.asarray(item["label"], dtype=bool).astype(np.float32)
        label = torch.from_numpy(label)
        output["label"] = label
        return output

    def _load_img(self, path) -> Image:
        img = Image.open(path)
        img = img.convert(mode="L")
        #img = img.convert(mode="RGB")
        return img    

    def _process_view(self, img):
        img = np.asarray(img, dtype=np.float32)
        if self.transform:
            img = self.perform_augmentation(img) 

    def perform_augmentation(self, img):
        #img = img.convert("RGB") # Ensure image is in RGB mode before augmentation
        img = img.astype(np.uint8)
        img = self.img_aug(image=img)
        img = img.astype(np.float32)
        return img


    def initialize_data(self, path) -> None:
        with open(path, newline='') as csvfile:
            # first row (header) will be treated as field names
            reader = csv.DictReader(csvfile, fieldnames=None)
            for row in reader:
                item = dict()
                item["label"] = []
                for view in self.views:
                    item[view] = row[view]
                for label in self.labels:
                    if label[0] == "~":
                        item["label"].append(1.0 - float(row[label[1:]]))
                        continue
                    item["label"].append(float(row[label]))
                item["id"] = len(self.data)
                self.data.append(item)
        random.shuffle(self.data)
        random.shuffle(self.data)
    
if __name__=="__main__":
    views = ["file_name"]
    # Order matter for labels
    labels = ["label", "~label"]
    ROOT_DIR = "/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Cable"
    csv_path = os.path.join(ROOT_DIR, "train.csv")
    mv_dst = augmentation(csv_path, views, labels, base_dir=ROOT_DIR, normalize=False)
    print(mv_dst.data[0])
    item = mv_dst[0]
    view1 = item[views[0]]
    print(view1.shape)
    view1 = view1.numpy()
    print(view1.shape)
    view1 = view1.reshape((256, 256, 3))
    view1 = view1.astype(np.uint8)

    img = Image.fromarray(view1)

    img = img.convert("RGB")
    img.save("cover.jpg")   