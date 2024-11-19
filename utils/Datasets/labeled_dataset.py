import os
import cv2
import numpy as np
import random

import torch
from torch.utils import data
from torchvision import transforms

from utils.utils import read_list
from baseline_models.UniFuse.datasets.util import Equirec2Cube


class LabelDataset(data.Dataset):
    """The Matterport3D Dataset"""

    def __init__(
        self, 
        root_dir, 
        list_file, 
        height=512, 
        width=1024, 
        disable_color_augmentation=False,
        disable_LR_filp_augmentation=False, 
        disable_yaw_rotation_augmentation=False, 
        is_training=False,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        rand_gamma=False,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        device='cpu',
        need_cube=False,
        ):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir
        self.rgb_depth_list = np.array(read_list(list_file))

        self.w = width
        self.h = height

        self.max_depth_meters = 10.0

        self.color_augmentation = not disable_color_augmentation
        self.LR_filp_augmentation = not disable_LR_filp_augmentation
        self.yaw_rotation_augmentation = not disable_yaw_rotation_augmentation

        self.is_training = is_training

        if self.color_augmentation:
            self.brightness = brightness
            self.contrast = contrast
            self.saturation = saturation
            self.hue = hue
            self.color_aug= transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)
        self.device = device

        # CUBE
        self.need_cube = need_cube
        if self.need_cube:
            self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)

        # gamma
        self.rand_gamma = rand_gamma

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][0])
        inputs['rgb_name'] = rgb_name
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        # -1 is unchanged format
        gt_depth = cv2.imread(depth_name, -1)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if depth_name.endswith('exr') or depth_name.endswith('.npy'):
            # EXR and NUMPY are 32 bit floatings, no need to adjust
            pass
        else:
            # Matterport3D: divide by 4000
            # Stanford2D3D: divide by 512
            # Structured3D: divide by 1000
            # Gibson: divide by 1000
            if 'st3d' in depth_name or 'gibson' in depth_name:
                gt_depth = gt_depth.astype(np.float32) / 1000
            elif 'sf3d' in depth_name or 'stanford' in depth_name:
                gt_depth = gt_depth.astype(np.float32) / 512
            else:
                gt_depth = gt_depth.astype(np.float32) / 4000
        gt_depth[gt_depth > self.max_depth_meters+1] = self.max_depth_meters + 1


        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)

        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))
        else:
            aug_rgb = rgb

        if self.need_cube:
            cube_rgb = self.e2c.run(rgb)
            cube_aug_rgb = self.e2c.run(aug_rgb)

            cube_rgb = self.to_tensor(cube_rgb.copy())
            cube_aug_rgb = self.to_tensor(cube_aug_rgb.copy())
            inputs["cube_rgb"] = cube_rgb
            inputs["normalized_cube_rgb"] = self.normalize(cube_aug_rgb)

        rgb = self.to_tensor(rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())
        if self.rand_gamma:
            p = np.random.uniform(1, 1.2)
            if np.random.randint(2) == 0:
                p = 1 / p
            aug_rgb = aug_rgb ** p

        inputs["rgb"] = rgb
        inputs["normalized_rgb_noaug"] = self.normalize(rgb)
        inputs["normalized_rgb"] = self.normalize(aug_rgb)


        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        inputs["val_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters)
                                & ~torch.isnan(inputs["gt_depth"]))

        inputs["gt_metric_depth"] = inputs["gt_depth"].clone()
        inputs["gt_depth"][inputs["val_mask"]] = 1 / inputs["gt_depth"][inputs["val_mask"]]
        inputs["gt_depth"] *= inputs["val_mask"].float()
        if inputs["gt_depth"][inputs["val_mask"]].any():
            inputs["gt_depth"][inputs["val_mask"]] -= inputs["gt_depth"][inputs["val_mask"]].min()
            inputs["gt_depth"][inputs["val_mask"]] /= inputs["gt_depth"][inputs["val_mask"]].max()

        return inputs
