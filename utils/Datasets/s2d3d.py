import os
import cv2
import numpy as np
import random

import torch
from torchvision import transforms

from utils.Datasets import LabelDataset


class Stanford2D3D(LabelDataset):
    """Unlabeled Dataset"""

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
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        relative=False,
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
        super().__init__(
            root_dir=root_dir, 
            list_file=list_file, 
            height=height, 
            width=width, 
            disable_color_augmentation=disable_color_augmentation,
            disable_LR_filp_augmentation=disable_LR_filp_augmentation, 
            disable_yaw_rotation_augmentation=disable_yaw_rotation_augmentation, 
            is_training=is_training,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            mean=mean,
            std=std,
            relative=relative,
            device=device,
            need_cube=need_cube,
        )
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][0])
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        # -1 is unchanged format
        gt_depth = cv2.imread(depth_name, -1)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if depth_name.endswith('exr') or depth_name.endswith('.npy'):
            pass
        else:
            gt_depth = gt_depth.astype(np.float32)/512
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
            #cube_rgb, cube_gt_depth = self.e2c.run(rgb, gt_depth[..., np.newaxis])
            cube_rgb = self.e2c.run(rgb)
            cube_aug_rgb = self.e2c.run(aug_rgb)

            cube_rgb = self.to_tensor(cube_rgb.copy())
            cube_aug_rgb = self.to_tensor(cube_aug_rgb.copy())
            inputs["cube_rgb"] = cube_rgb
            inputs["normalized_cube_rgb"] = self.normalize(cube_aug_rgb)

        rgb = self.to_tensor(rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())

        inputs["rgb"] = rgb
        inputs["normalized_rgb_noaug"] = self.normalize(rgb)
        inputs["normalized_rgb"] = self.normalize(aug_rgb)


        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        inputs["val_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters)
                                & ~torch.isnan(inputs["gt_depth"]))

        if self.relative:
            inputs["gt_metric_depth"] = inputs["gt_depth"].clone()
            inputs["gt_depth"][inputs["val_mask"]] = 1 / inputs["gt_depth"][inputs["val_mask"]]
            inputs["gt_depth"][inputs["val_mask"]] -= inputs["gt_depth"][inputs["val_mask"]].min()
            inputs["gt_depth"][inputs["val_mask"]] /= inputs["gt_depth"][inputs["val_mask"]].max()
        
        """
        cube_gt_depth = torch.from_numpy(np.expand_dims(cube_gt_depth[..., 0], axis=0))
        inputs["cube_gt_depth"] = cube_gt_depth
        inputs["cube_val_mask"] = ((cube_gt_depth > 0) & (cube_gt_depth <= self.max_depth_meters)
                                   & ~torch.isnan(cube_gt_depth))
        """

        return inputs