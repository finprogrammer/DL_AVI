import csv
import os
import random
import imgaug.augmenters as ia
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from config_loader import load_config

mode = "train_2"  # Change this to "debug" when debugging
CONFI = load_config(mode)
    
datatest = []
dataval = []
class MultiViewDataset(object):
    def __init__(
        self,
        path,
        views,
        labels,
        base_dir=None,
        transform=True,
        normalize=True,
        pil_image_mode="L",
        Test=False,
        Val=False,
        Feature_extraction=False,
    ) -> None:
        self.data = []
        self.transform = transform
        self.normalize = normalize
        self.Feature_extraction = Feature_extraction
        self.views = views
        self.labels = labels
        self.Test = Test
        self.Val = Val
        self.base_dir = base_dir
        self.initialize_data(path)
        self.pil_image_mode = (
            pil_image_mode  # "'RGB' for rgb image and 'L' for grayscale image"
        )
        self.image_size = (CONFI['W'], CONFI['H'])  # (300, 300)#
        self.img_aug = ia.Sometimes(
            1,
            ia.SomeOf(
                1,
                [
                    ia.Fliplr(0.8),
                    ia.Flipud(0.8),
                    ia.Affine(
                        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
                    ),
                    ia.Affine(scale=(0.7, 1.1)),
                    ia.Affine(rotate=(-45, 45)),
                    # ia.CenterCropToFixedSize(height=100, width=100)
                    # ia.CoarseDropout((0.0, 0.8), size_percent=(0.02, 0.25))
                    # ia.Affine(shear=(-16, 16)),
                ],
            ),
        )
        self.img_fd = ia.Sequential(
            [
                # ia.pillike.FilterFindEdges(),
                # ia.pillike.FilterEdgeEnhance(),
                ia.pillike.FilterFindEdges(),
                # ia.pillike.FilterContour(),
            ]
        )
        self.img_aug_tensor = T.Compose(
            [
                T.ToTensor(),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                #T.RandomAffine(degrees=(-45, 45), translate=(0.1, 0.2), scale=(0.8, 1.3)),#cable_ip
                #T.RandomAffine(degrees=(-45, 45), translate=(0.2, 0.27), scale=(0.8, 1.3)),#smp_ip
                #T.RandomAffine(degrees=(-45, 45), translate=(0.2, 0.2), scale=(0.8, 1.3)),#cover_ip
                #T.RandomAffine(degrees=(-45, 45), translate=(0.45, 0.45), scale=(0.8, 1.3)),#wh_ip

                T.RandomAffine(degrees=(-45, 45), translate=(0.2, 0.16), scale=(0.8, 1.3)),#w_h_no_preprcesing
                #T.RandomAffine(degrees=(-45, 45), translate=(0.05, 0.05), scale=(0.7, 1.1)),
                # T.RandomAffine(degrees=(-45, 45), translate=(0.3, 0.1), scale=(0.8, 1.5)),#WH
                #T.RandomAffine(degrees=(-25, 25), translate=(0.5, 0.1), scale=(0.8, 1.8)),#sheetmetal package
                #T.RandomAffine(degrees=(-75, 75), translate=(0.2, 0.1), scale=(0.8, 1.0)),#cover
            ]
        )
        self.transform_image = T.Compose(
            [T.ToTensor(), T.Resize(224), T.CenterCrop(224), T.Normalize([0.5], [0.5])]
        )

    def __getitem__(self, i) -> dict:
        item = self.data[i]
        output = {}
        for view in self.views:
            img = item[view]
            if self.base_dir is not None:
                img = os.path.join(self.base_dir, img)
            if CONFI['Model'] == "Dinov2":
                output[view] = self.load_dinov2_image(img)
            elif CONFI['Model'] == "swin_v2_b":
                output[view] = self.load_swin_image(img)
            elif CONFI['Model'] == "resnext50_32x4d":
                output[view] = self.load_resnext_image(img)
            else:
                output[view] = self.get_image(img)

        label = np.asarray(item["label"], dtype=np.float32)
        label = torch.from_numpy(label)
        output["label"] = label
        output["image_name_string"] = item["image_name_string"]
        return output
    
    def get_image(self, path):
        img = Image.open(path)
        
        if self.Feature_extraction:
            img = self._Feature_extraction(img)

        img = self.pre_process_img(img)

        if CONFI['Model'] == "swin_v2_b":
            swin_transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(272, interpolation=InterpolationMode.BICUBIC),
                T.CenterCrop(256),
            ]
        )
            img = swin_transform(img)
        elif CONFI['Model'] == "resnext50_32x4d":
            resize_size = 232
            crop_size = 224
            res_transform = T.Compose(
            [
                T.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
                T.CenterCrop(crop_size),
            ]
            )
            img = res_transform(img)

        if self.transform:
            img = self.img_aug_tensor(img)        

        if self.normalize:
            img = self.ar_nor(img)

        if self.pil_image_mode == "L":
            img = T.functional.to_grayscale(img)
            img = img.expand(3, -1, -1)

        return img        


    def load_dinov2_image(self, path):
        img = Image.open(path)

        # if self.Feature_extraction:
        #     img = self._Feature_extraction(img)

        # new_w, new_h = self.image_size
        # background = Image.new("RGB", (new_w, new_h))
        # img.thumbnail((new_w - 5, new_h - 5))
        # center_background = (new_w // 2, new_h // 2)
        # center_img = (img.width // 2, img.height // 2)
        # paste_position = (
        # center_background[0] - center_img[0],
        # center_background[1] - center_img[1]
        # )
        # background.paste(img, paste_position)
        # img = background
        output_size = (378, 378)
        resize = T.Compose(
            [
            T.Resize(output_size),
            ]
        )
        img = resize(img)

        if self.transform:
            data_transforms = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    #T.RandomAffine(degrees=(-45, 45), translate=(0.2, 0.16), scale=(0.8, 1.3)),#w_h
                    T.RandomAffine(degrees=(-45, 45), translate=(0.05, 0.05), scale=(0.8, 1.0)),#cover_ip
                ]
            )
            img = data_transforms(img)
            
        if self.normalize:
            nor = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            img = nor(img)       
        return img
    
    def ar_nor(self,img):
        t_transform = T.Compose([T.ToTensor()])
        img = t_transform(img)        
        mean = img.mean([1, 2])
        std = img.std([1, 2])
        if (std == 0).all():
            print("Standard deviation is zero along all dimensions.")
            # Add a small epsilon to avoid division by zero
            std += 1e-6

        # normalize_transform= T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        normalize_transform = T.Compose([T.Normalize(mean, std)])
        img = normalize_transform(img)
        return img

    def load_swin_image(self, path):
        img = Image.open(path)
        if self.Feature_extraction:
            img = self._Feature_extraction(img)
        #img = self.pre_process_img(img)
        swin_transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(271, interpolation=InterpolationMode.BICUBIC),
                T.CenterCrop(270),
            ]
        )
        img = swin_transform(img)
        if self.transform:
            img = self.img_aug_tensor(img)
        img = self.ar_nor(img)    
        if self.pil_image_mode == "L":
            img = T.functional.to_grayscale(img)
            img = img.expand(3, -1, -1) 
        return img

    def pre_process_img(self, img):
        new_w, new_h = self.image_size
        background = Image.new("RGB", (new_w, new_h))
        img.thumbnail((new_w - 5, new_h - 5))
        center_background = (new_w // 2, new_h // 2)
        center_img = (img.width // 2, img.height // 2)
        paste_position = (
            center_background[0] - center_img[0],
            center_background[1] - center_img[1],
        )
        background.paste(img, paste_position)
        img = background
        return img

    def load_resnext_image(self, path):
        img = Image.open(path)
        if self.Feature_extraction:
            img = self._Feature_extraction(img)
        img = self.pre_process_img(img)
        # # Resize and crop
        # resize_size = 232
        # crop_size = 224
        # res_transform = T.Compose(
        #     [
        #         T.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
        #         T.CenterCrop(crop_size),
        #     ]
        # )
        # img = res_transform(img)
        # img = T.ToTensor()(img)
        # # Normalize
        # img = self.ar_nor(img)
        if self.transform:
            img = self.img_aug_tensor(img)
        if self.pil_image_mode == "L":
            img = T.functional.to_grayscale(img)
            img = img.expand(3, -1, -1)
        return img

    def __len__(self):
        return len(self.data)

    def _load_img(self, path) -> Image:
        img = Image.open(path)
        if self.Feature_extraction:
            img = self._Feature_extraction(img)
        img = img.convert(mode=self.pil_image_mode)
        # img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        # img = img.filter(ImageFilter.FIND_EDGES)
        # img = img.filter(ImageFilter.CONTOUR)
        return img

    def initialize_data(self, path) -> None:
        with open(path, newline="") as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=None)
            for row in reader:
                item = dict()
                item["label"] = []
                item["image_name_string"] = []
                for view in self.views:
                    item[view] = row[view]
                item["image_name_string"] = row[view]
                if CONFI['Model'] == "Dinov2":
                    for label in self.labels:
                        item["label"] = row["label"]                
                else:
                    for label in self.labels:
                        if label[0] == "~":
                            item["label"].append(1.0 - float(row[label[1:]]))
                            continue
                        item["label"].append(float(row[label]))
                item["id"] = len(self.data)
                self.data.append(item)
                if self.Test: # lists for plotting misclassified samples
                    datatest.append(item)
                if self.Val:
                    dataval.append(item)
        if not self.Test and not self.Val:
            random.shuffle(self.data)
            random.shuffle(self.data)

    def perform_augmentation(self, img):
        img = self.img_aug(image=img)
        return img
        
    def _Feature_extraction(self, img):
        img = np.asarray(img, dtype=np.uint8)
        img = self.img_fd(image=img)
        img = Image.fromarray(np.asarray(img))
        return img

if __name__ == "__main__":

    # import CONFI
    views = ["file_name"]
    # Order matter for labels
    labels = ["label", "~label"]
    ROOT_DIR = "/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Sheet_Metal_Package"
    csv_path = os.path.join(ROOT_DIR, "test.csv")
    print(datatest)
    mv_dst = MultiViewDataset(
        csv_path,
        views,
        labels,
        base_dir=ROOT_DIR,
        transform=True,
        Test=True,
        Val=True,
        normalize=True,
        pil_image_mode="RGB",
        Feature_extraction=False,
    )

    print(mv_dst.data[0])

    item = mv_dst[500]
    view1 = item[views[0]]
    print(f"nor_std_aug = {view1.std(dim=(1, 2))}")
    print(f"nor_mean_aug = {view1.mean(dim=(1, 2))}")
    img_n = view1.numpy()
    print(view1.shape)
    min_value = np.min(img_n)
    max_value = np.max(img_n)
    print(f"Min Pixel Value after nor: {min_value}")
    print(f"Max Pixel Value after nor: {max_value}")

    img_n = view1.numpy()
    max_value = np.max(img_n)
    img = view1 / max_value
    transform_to_pil = T.Compose([T.ToPILImage()])
    img = transform_to_pil(img)
    img.save("cover_pil_new.jpg")
