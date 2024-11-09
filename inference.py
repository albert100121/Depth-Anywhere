import argparse
import os
from typing import List

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from baseline_models.UniFuse.networks import UniFuse
from baseline_models.BiFuseV2 import BiFuse
import sys
sys.path.append("baseline_models/HoHoNet/")
from baseline_models.HoHoNet.lib.model.hohonet import HoHoNet
sys.path.append("baseline_models/EGformer/")
from baseline_models.EGformer.models.egformer import EGDepthModel
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
        default='data/examples/sf3d', 
        help='Path to the input directory.')
    parser.add_argument(
        '--pretrained_weight', 
        default='checkpoints/UniFuse/UniFuse_SpatialAudioGen.pth', 
        help='Path to the checkpoint.')
    parser.add_argument(
        '--output_dir', 
        default='outputs/', 
        help='Path to the output directory.')
    args = parser.parse_args()

    # model name
    args.baseline_model = os.path.split(
        os.path.split(args.pretrained_weight)[0]
        )[1]

    # append output dir
    data_name = os.path.basename(os.path.dirname(args.input_dir))
    args.output_dir = os.path.join(args.output_dir, data_name)
    ckpt_name = os.path.splitext(os.path.basename(args.pretrained_weight))[0]
    args.output_dir = os.path.join(args.output_dir, args.baseline_model, ckpt_name)

    args.requires_cube = False
    if args.baseline_model.upper() in ('UNIFUSE'):
        args.requires_cube = True

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f'{args.input_dir} not found!')

    return args


def load_data(input_dir: str) -> List[str]:
    """Load data from input dir."""

    data_extensions = ['.png', '.jpg']
    # get data name
    data_names = [x for x in os.listdir(
        input_dir) if os.path.splitext(x)[-1] in data_extensions]

    return data_names


def load_model(ckpt_path: str, device: str, model_name: str='UniFuse'):
    """Load pretrained model."""

    print(f"Load baseline model: {model_name}'")
    if model_name.upper() == 'UNIFUSE':
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
    elif model_name.upper() == 'BIFUSEV2':
        # set arguments
        dnet_args = {
            'layers': 34,
            'CE_equi_h': [8, 16, 32, 64, 128, 256, 512]
        }
        model = BiFuse.SupervisedCombinedModel('outputs', dnet_args)
    elif model_name.upper() == 'HOHONET':
        model = HoHoNet(
            emb_dim=256,
            backbone_config={
                'module': 'Resnet',
                'kwargs': {
                    'backbone': 'resnet50'}
            },
            decode_config={
                'module': 'EfficientHeightReduction'},
            refine_config={
                'module': 'TransEn',
                'kwargs': {
                    'position_encode': 256,
                    'num_layers': 1
                    }
            },
            modalities_config={
                'DepthEstimator': {
                    'basis': 'dct',
                    'n_components': 64,
                    'loss': 'l1'
                    }
                }
        )
        model.forward = model.infer
    elif model_name.upper() == 'EGFORMER':
        model = EGDepthModel(hybrid=False)
    else:
        raise NotImplementedError(f'Baseline model {model_name} not implemented!')
    
    # to device
    model.to(device)

    # load pretrained weight
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model.eval()
    
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
    model = load_model(args.pretrained_weight, device, args.baseline_model)
    model.eval()

    for name in tqdm(data_names, desc='run model images'):
        path = os.path.join(args.input_dir, name)
        output_path = os.path.join(args.output_dir, name)

        rgb, cube = load_rgb(path, device=device)
        with torch.no_grad():
            if args.requires_cube:
                outputs = model(rgb, cube)
            else:
                outputs = model(rgb)

        depth = outputs['pred_depth'].squeeze().cpu().numpy()
        depth = depth - depth.min()
        depth = depth / depth.max()
        depth = (depth * 255).astype(np.uint8)
        print(output_path)
        cv2.imwrite(output_path, depth)
    

if __name__ == '__main__':
    main()