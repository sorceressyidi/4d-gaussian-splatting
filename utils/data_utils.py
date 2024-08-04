import os
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np

# Related to depth supervison -- add depth gt pictures
class CameraDataset(Dataset):
    
    def __init__(self, viewpoint_stack, white_background):
        self.viewpoint_stack = viewpoint_stack
        self.bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        
    def __getitem__(self, index):
        viewpoint_cam = self.viewpoint_stack[index]
        if viewpoint_cam.meta_only:
            with Image.open(viewpoint_cam.image_path) as image_load:
                im_data = np.array(image_load.convert("RGBA"))
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + self.bg * (1 - norm_data[:, :, 3:4])
            image_load = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            resized_image_rgb = PILtoTorch(image_load, viewpoint_cam.resolution)
            viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
            if resized_image_rgb.shape[1] == 4:
                gt_alpha_mask = resized_image_rgb[3:4, ...]
                viewpoint_image *= gt_alpha_mask
            else:
                viewpoint_image *= torch.ones((1, viewpoint_cam.image_height, viewpoint_cam.image_width))
        else:
            viewpoint_image = viewpoint_cam.image
        if viewpoint_cam.depth is not None:
            depth_image = viewpoint_cam.depth
        else:
            depth_image = None
        if viewpoint_cam.error is not None:
            error = viewpoint_cam.error
        else:
            error = None

        '''
            norm_data = depth_image / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + self.bg * (1 - norm_data[:, :, 3:4])
            depth_image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            resized_image_rgb = PILtoTorch(depth_image, viewpoint_cam.resolution)
            depth_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
            if resized_image_rgb.shape[1] == 4:
                gt_alpha_mask = resized_image_rgb[3:4, ...]
                depth_image *= gt_alpha_mask
            else:
                depth_image *= torch.ones((1, viewpoint_cam.image_height, viewpoint_cam.image_width))
        else:
            depth_image = None   
        '''
        return viewpoint_image, viewpoint_cam, depth_image,error
    
    def __len__(self):
        return len(self.viewpoint_stack)
    
