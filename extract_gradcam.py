import os
import numpy as np
import torch
import cv2
from torchvision.transforms import v2 as T
import monai
from PIL import Image

# from extract_gradcam import grad_cam


from config_loader import load_config
mode = "train_2"  # Change this to "debug" when debugging
CONFI = load_config(mode)

# instance.grad_image



class grad_cam(object):
    def __init__(self, iter = None, x2=None, Gradcam_save_dir=None, val_batch=None, logger_version=None, backbone = None, match_indices = None, ROOT_DIR = None):
        self.backbone = backbone
        self.x2 = x2
        self.Gradcam_save_dir = Gradcam_save_dir
        self.val_batch = val_batch
        self.gradcam = monai.visualize.GradCAM(
            nn_module=self.backbone,
            target_layers="features.7.0",  # self.backbone.features[7][2]
        )
        self.logger_version = logger_version
        self.iter = iter
        self.match_indices = match_indices
        self.ROOT_DIR = ROOT_DIR

    def grad_image(self):
        for i in range(len(self.match_indices[0])): 
            j = self.match_indices[0][i]
            #iter = self.iter
            folder = self.Gradcam_save_dir
            with torch.set_grad_enabled(True):
                    raw_image = self.x2[j]
                    raw_image = raw_image.squeeze(0)
                    ###segmentation mask for class1
                    gradcam_img2 = self.gradcam(x=raw_image[None], class_idx=None)
                    view1 = gradcam_img2
                    view1 = view1.squeeze(0)
                    img_n = view1.cpu().numpy()
                    print(view1.shape)
                    max_value = np.max(img_n)
                    img_n = view1.cpu().numpy()
                    max_value = np.max(img_n)
                    img = view1 / max_value
                    transform_to_pil = T.Compose([T.ToPILImage()])
                    img = transform_to_pil(img)
                    filename = self.val_batch["image_name_string"]
                    filename = filename[j]
                    filename = filename.replace('images/', '')
                    filename = filename.replace('.jpg', '')
                    filename = f"class1__{self.logger_version}__{filename}.jpg"#_{iter}
                    save_path_class_1 = os.path.join(folder, filename)
                    img.save(save_path_class_1)
                    

                    ###segmentation mask for class0
                    gradcam_img = self.gradcam(x=raw_image[None], class_idx=0)
                    view1 = gradcam_img
                    view1 = view1.squeeze(0)
                    img_n = view1.cpu().numpy()
                    print(view1.shape)
                    max_value = np.max(img_n)
                    img_n = view1.cpu().numpy()
                    max_value = np.max(img_n)
                    img = view1 / max_value
                    transform_to_pil = T.Compose([T.ToPILImage()])
                    img = transform_to_pil(img)
                    filename = self.val_batch["image_name_string"]
                    filename = filename[j]
                    filename = filename.replace('images/', '')
                    filename = filename.replace('.jpg', '')                                    
                    filename = f"class0__{self.logger_version}__{filename}.jpg"
                    save_path_class_0 = os.path.join(folder, filename)
                    img.save(save_path_class_0)

                    ###save original image
                    filename = self.val_batch["image_name_string"]
                    filename = filename[j]
                    path = os.path.join(self.ROOT_DIR, filename)
                    img = Image.open(path)
                    new_w, new_h = (CONFI['W'], CONFI['H'])
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
                    filename = filename.replace('images/', '')
                    filename = filename.replace('.jpg', '')                                  
                    filename = f"original_image__{self.logger_version}__{filename}.jpg"
                    save_path_img = os.path.join(folder, filename)
                    img.save(save_path_img)

                    ###combine original and the gradcam image for class 0
                    original_image = cv2.imread(save_path_img)
                    cam_image = cv2.imread(save_path_class_0)
                    original_image = cv2.convertScaleAbs(original_image)
                    cam_image = cv2.convertScaleAbs(cam_image)
                    heatmap = cv2.applyColorMap(cam_image, cv2.COLORMAP_RAINBOW)
                    heatmap = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)
                    filename = self.val_batch["image_name_string"]
                    filename = filename[j]
                    filename = filename.replace('images/', '')
                    filename = filename.replace('.jpg', '')                    
                    filename = f"gradcam_0__{self.logger_version}__{filename}.jpg"
                    save_path = os.path.join(folder, filename)
                    cv2.imwrite(save_path, heatmap)
                    
                    ###combine original and the gradcam image for class 1
                    original_image = cv2.imread(save_path_img)
                    cam_image = cv2.imread(save_path_class_1)
                    original_image = cv2.convertScaleAbs(original_image)
                    cam_image = cv2.convertScaleAbs(cam_image)
                    heatmap = cv2.applyColorMap(cam_image, cv2.COLORMAP_RAINBOW)
                    heatmap = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)
                    filename = self.val_batch["image_name_string"]
                    filename = filename[j]
                    filename = filename.replace('images/', '')
                    filename = filename.replace('.jpg', '')                    
                    filename = f"gradcam_1__{self.logger_version}__{filename}.jpg"
                    save_path = os.path.join(folder, filename)
                    cv2.imwrite(save_path, heatmap)
                    i += 1 
