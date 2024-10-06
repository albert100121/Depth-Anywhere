import argparse
import os
from typing import List

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from baseline_models.UniFuse.networks import UniFuse
from utils.Projection import py360_E2C


np.bool = np.bool_
np.float = np.float32
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_args() -> argparse.Namespace:
    """Load input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', 
        default='data/examples/', 
        help='Path to the input directory.')
    parser.add_argument(
        '--pretrained_weight', 
        default='checkpoints/UniFuse/UniFuse_SpatialAudioGen.pth', 
        help='Path to the checkpoint.')
    parser.add_argument(
        '--output_dir', 
        default='outputs/examples', 
        help='Path to the output directory.')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError

    return args


def load_data(input_dir: str) -> List[str]:
    """Load data from input dir."""

    data_extensions = ['.png', '.jpg']
    # get data name
    data_names = [x for x in os.listdir(
        input_dir) if os.path.splitext(x)[-1] in data_extensions]

    return data_names



def load_model(ckpt_path: str, device: str):
    """Load pretrained model."""

    # set arguments
    model_dict = {
        'num_layers': 18,
        'equi_h': 512,
        'equi_w': 1024,
        'pretrained': True,
        'max_depth': 10.0,
        'fusion_type': 'cee',
        'se_in_fusion': True
    }
    model = UniFuse(**model_dict)
    model.to(device)

    # load pretrained weight
    
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    
    return model


def load_rgb(
        path: str, 
        height: int=512, 
        width: int=1024, 
        device: str='cuda') -> torch.tensor:
    """Load rgb image and return processed tensor."""

    # Define Function
    totensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    E2C = py360_E2C(equ_h=height, equ_w=width, face_w=(height//2))

    # load rgb
    rgb = cv2.imread(path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # process size
    rgb = cv2.resize(
        rgb, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    cube_rgb = E2C.run(rgb)

    # to tensor
    tensor_rgb = totensor(rgb)
    normalize_rgb = normalize(tensor_rgb)
    tensor_cube_rgb = totensor(cube_rgb)
    normalize_cube_rgb = normalize(tensor_cube_rgb)

    # CHW -> BCHW
    normalize_rgb = normalize_rgb.unsqueeze(0)
    normalize_cube_rgb = normalize_cube_rgb.unsqueeze(0)

    # to device
    normalize_rgb = normalize_rgb.to(device)
    normalize_cube_rgb = normalize_cube_rgb.to(device)

    return normalize_rgb, normalize_cube_rgb


def main():
    args = get_args()
    # create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_names = load_data(args.input_dir)
    model = load_model(args.pretrained_weight, device)
    model.eval()

    for name in tqdm(data_names, desc='run model images'):
        path = os.path.join(args.input_dir, name)
        output_path = os.path.join(args.output_dir, name)

        rgb, cube = load_rgb(path, device=device)
        print(rgb.shape, cube.shape)
        with torch.no_grad():
            outputs = model(rgb, cube)
        depth = outputs['pred_depth'][0][0].cpu().numpy()
        print(depth.shape, depth.max(), depth.min())
        depth = depth - depth.min()
        depth = depth / depth.max()
        depth = (depth * 255).astype(np.uint8)
        cv2.imwrite(output_path, depth)
    

if __name__ == '__main__':
    main()