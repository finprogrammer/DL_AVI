# from pytorch_grad_cam.utils.image import show_cam_on_image
# import matplotlib.pyplot as plt
# image = "/home/vault/iwfa/iwfa048h/screw_img_20.jpg"
# grayscale_cam = "/home/vault/iwfa/iwfa048h/grad_cam/seg_mask_100.jpg"

# plt.figure(figsize=(12, 12))
# visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
# plt.imshow(visualization)
# #plt.title(f'pred: {pred:.1f} label: {label}')
# plt.axis('off')
# plt.savefig('overlay.jpg') 
# plt.show()



import cv2
import numpy as np

def show_cam_on_image(image_path, cam_path, save_path):
    # Load the original image and the CAM image
    original_image = cv2.imread(image_path)
    cam_image = cv2.imread(cam_path)

    # Ensure that the images are in the correct format (CV_8UC3)
    original_image = cv2.convertScaleAbs(original_image)
    heatmap = cv2.convertScaleAbs(cam_image)

    # Apply the CAM heatmap on the original image
    #heatmap = cv2.applyColorMap(cam_image, cv2.COLORMAP_JET)
    #heatmap = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

    #img = cv2.imread('./data/Elephant/data/05fig34.jpg')
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.8 + original_image
    #cv2.imwrite('./map.jpg', superimposed_img)

    # Save the heatmap to a file
    cv2.imwrite(save_path, superimposed_img)

# Paths to the original image and the CAM image
image_path = "/home/vault/iwfa/iwfa048h/grad_cam/cover_1/cover_img_560.jpg"
cam_path = "/home/vault/iwfa/iwfa048h/grad_cam/cover_1/cover_seg_mask_class0_560.jpg"
save_path = "/home/vault/iwfa/iwfa048h/grad_cam/heatmap_screw_40.jpg"  # Specify the path to save the heatmap image

# Generate and save the heatmap overlay on the original image
show_cam_on_image(image_path, cam_path, save_path)






# import os
# import cv2
# import numpy as np

# def show_cam_on_image(image_path, cam_path, save_path):
#     # Load the original image and the CAM image
#     original_image = cv2.imread(image_path)
#     cam_image = cv2.imread(cam_path)

#     # Ensure that the images are in the correct format (CV_8UC3)
#     original_image = cv2.convertScaleAbs(original_image)
#     cam_image = cv2.convertScaleAbs(cam_image)

#     # Apply the CAM heatmap on the original image
#     heatmap = cv2.applyColorMap(cam_image, cv2.COLORMAP_JET)
#     heatmap = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

#     # Save the heatmap to a file
#     cv2.imwrite(save_path, heatmap)

# # Folder containing the original images
# image_folder = "/home/vault/iwfa/iwfa048h/heatmap_debug"
# # Folder containing the CAM images
# cam_folder = "/home/vault/iwfa/iwfa048h/heatmap_debug_map"
# # Folder to save the heatmaps
# save_folder = "/home/vault/iwfa/iwfa048h/heatmap_debug_save"

# # List all files in the original images folder
# image_files = os.listdir(image_folder)

# # Iterate over each image file
# for image_file in image_files:
#     # Construct paths to the original image, CAM image, and save location
#     image_path = os.path.join(image_folder, image_file)
#     cam_path = os.path.join(cam_folder, image_file)  # Assuming CAM image names match original image names
#     save_path = os.path.join(save_folder, f"heatmap_{image_file}")  # Save heatmap with original image name prefix

#     # Generate and save the heatmap overlay on the original image
#     show_cam_on_image(image_path, cam_path, save_path)