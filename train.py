"""This is the training code for Depth Anywhere

Usage
    python train.py --config [Path to your config file]
"""
import argparse
import os
import sys
import random
from collections import defaultdict
from typing import Dict

import torch.utils
import torch.utils.data
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
#### future api and openexr ####
np.bool = np.bool_
np.float = np.float32

from utils import parse_args, get_model, get_optim, save_model, save_log, get_unlabel_data
from utils.metric import Affine_Inv_Evaluator
# sys.path.append("/project/albert/distill_pers/Depth_Anything_exp/")
# from Depth_Anything.depth_anything.dpt import DepthAnything
from foundation_models.depth_anything.dpt import DepthAnything
# sys.path.append("/project/albert/distill_pers/Depth_Anything_exp/utils/CE")
# from Depth_Anything_exp.utils.CE import Equirec2Cube, Cube2Equirec, EquirecDepth2Points
# from Depth_Anything_exp.utils.CE.Equirec2Cube import EquirecRotate2 as EquirecRotate
from utils.Projection.Cube2Equirec import Cube2Equirec
from utils.Projection.Equirec2Cube import Equirec2Cube
from utils.Projection import EquirecGrid as EG
from utils import Conversion

# sys.path.append("/project/albert/distill_pers/Depth_Anything_exp/utils/CE")
# from Depth_Anything_exp.utils.CE import Depth2Points, Equirec2Cube, Cube2Equirec
# sys.path.append("/project/albert/distill_pers/baseline_models/UniFuse/UniFuse")
# from baseline_models.UniFuse.UniFuse.networks.layers import Cube2Equirec as PanoC2E
# from baseline_models.UniFuse.UniFuse.datasets.util import Equirec2Cube as PanoE2C
# from utils.metric import affine_invariant
# BiFuseV2 Rotate
# sys.path.append("/project/albert/distill_pers/baseline_models/BiFusev2")
from utils.Projection import EquirecRotate as ER2


def init():
    # Init
    args, args_model = parse_args()
    args.device = torch.device('cpu' if args.no_cuda else 'cuda')
    args.model_name = args_model['model_setting']['model_name']

    # create folder
    save_folder = os.path.join(args.ckpt, args.model_name, args.id)
    os.makedirs(save_folder, exist_ok=True)
    # tensorboard
    log_path = os.path.join(args.log, args.model_name, args.id)
    writer = dict()
    for mode in ['train', 'val', 'test', 'zeroshot']:
        writer[mode] = SummaryWriter(os.path.join(log_path, mode))
    # open permission
    os.system(f'chmod -cR 777 {save_folder}')
    args.save_folder = save_folder

    # Random Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == torch.device('cuda'):
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.pth:
        ckpt = torch.load(args.pth)
        model_ckpt = optim_ckpt = settings = None
        if 'model' in ckpt.keys():
            model_ckpt = ckpt['model']
            optim_ckpt = ckpt['optim']
            settings = ckpt['settings']
        else:
            model_ckpt = ckpt
    else:
        model_ckpt = optim_ckpt = settings = None

    # BIFUSE V2
    if args.model_name.upper() == 'BIFUSEV2':
        args_model['model_kwargs']['save_path'] = args.save_folder

    model = get_model(args_model['model_setting']['model_name'], args.device, model_ckpt, dict(args_model['model_kwargs']))
    optimizer = get_optim(args, model, optim_ckpt)

    return args, model, optimizer, writer, settings


class ProcessDABatch():
    def __init__(self, h, w, CUDA=True, rot='v1'):
        self.h = h
        self.w = w
        self.device = 'cuda' if CUDA else 'cpu'
        self.BiFuse_C2E = Cube2Equirec(h//2, h)
        # self.d2p = EG.to_xyz
        self.EG = EG()
        self.BiFuse_E2C = Equirec2Cube(cube_dim=h//2, equ_h=h, FoV=90)
        if CUDA:
            self.BiFuse_C2E = self.BiFuse_C2E.cuda()
            self.BiFuse_E2C = self.BiFuse_E2C.cuda()
        # Equi Rotate
        self.ER = ER2(h)

    def process_output(self, inputs, outputs):
        gt = inputs["gt_depth"]
        gt_cube = inputs["pseudo_depth"]    # B1, B2, B3...
        mask_cube = inputs["pseudo_mask"]
        equi_batch = gt.shape[0]
        cube_batch = gt_cube.shape[0]

        pred_equi_disp = outputs["pred_depth"][equi_batch:].clone()     # call by value with gradient
        # by fuse doesn't have sigmoid
        shift = pred_equi_disp.min()
        if shift < 0:
            pred_equi_disp = pred_equi_disp - shift
        pred_equi_disp = self.rotate(pred_equi_disp, self.rot_vec, mode='nearest')

        # Pseudo Label
        if mask_cube is None:
            mask_cube = gt_cube > 0

        """
        To CUBE
        """
        # equi_disp -> equi_depth
        pred_equi_disp[pred_equi_disp != 0] = 1 / pred_equi_disp[pred_equi_disp != 0]
        # rename with reference
        pred_equi_depth = pred_equi_disp
        # depth to points
        points = self.EG.to_xyz(pred_equi_depth)   # B, 3, H, W
        # get cube_x, cube_y, cube_z
        # B, D, F, L, R, U (face)
        # -z, y, z, -x, x, -y (axis)
        pred_depth_cube_x = self.BiFuse_E2C(points[:, 0].unsqueeze(1), 'nearest')   # B, H, W -> B, 1, H, W -> 6B, 
        pred_depth_cube_y = self.BiFuse_E2C(points[:, 1].unsqueeze(1), 'nearest')
        pred_depth_cube_z = self.BiFuse_E2C(points[:, 2].unsqueeze(1), 'nearest')
        # L, R -> -x, x
        # D, U -> y, -y
        # B, F -> -z, z
        pred_depth_cube = pred_depth_cube_x
        for i in range(0, cube_batch, 6):
            pred_depth_cube[i] = -pred_depth_cube_z[i]
            pred_depth_cube[i+1] = pred_depth_cube_y[i+1]
            pred_depth_cube[i+2] = pred_depth_cube_z[i+2]
            pred_depth_cube[i+3] = -pred_depth_cube_x[i+3]
            pred_depth_cube[i+4] = pred_depth_cube_x[i+4]
            pred_depth_cube[i+5] = -pred_depth_cube_y[i+5]

        # depth 2 disp
        pred_depth_cube[pred_depth_cube != 0] = 1 / pred_depth_cube[pred_depth_cube != 0]

        outputs["pred_depth_cube"] = pred_depth_cube    # assigne cube disp

        return inputs, outputs

    def randRotate(self, equi):
        """
        equi: torch.tensor
        """
        self.rot_vec = (torch.rand(3) - 0.5) * 90

        equi = self.rotate(equi, self.rot_vec)
        
        return equi, self.rot_vec

    def rotate(self, equi, rot_vec, euler_R_ref=None, mode:str='bilinear'):
        angle = rot_vec / 180 * np.pi
        angle = angle.reshape([1, 3]).cuda()
        import pytorch3d.transforms.rotation_conversions as p3dr
        euler_R_ref = p3dr.euler_angles_to_matrix(angle, convention='XYZ')
        equi = self.ER(equi, rotation_matrix=euler_R_ref.transpose(1, 2), mode=mode)

        return equi


def train_joint_unlabel(
        args: argparse.ArgumentParser.parse_args,
        model: nn.Module, 
        DA: nn.Module, 
        iter_loader: enumerate, 
        label_loader: DataLoader, 
        optimizer: optim, 
        epoch: int, 
        writer: SummaryWriter, 
        unlabel_loader: torch.utils.data.DataLoader, 
        ProB: ProcessDABatch):
    """Train function for joint label/pseudo label."""
    DA = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format('vitl')).to(args.device).eval()
    model.train()
    total_loss = defaultdict(float)

    # label first
    pbar = tqdm(label_loader)
    pbar.set_description("Training Epoch_{} on label".format(epoch))
    
    for batch_idx, inputs in enumerate(pbar):
        """
        Depth anything 
        """
        iter_idx, unlabel = next(iter_loader)
        if iter_idx == len(unlabel_loader) - 1:
            # reset label loader
            iter_loader = enumerate(unlabel_loader)
        
        outputs = process_both(args, batch_idx, inputs, unlabel, len(label_loader), ProB, DA, model, optimizer, total_loss)
        
    save_log(writer, inputs, outputs, total_loss, args)
    return iter_loader


def process_both(
        args: argparse.ArgumentParser.parse_args,
        batch_idx: int,
        inputs: Dict[str, torch.tensor],
        unlabel: torch.utils.data.DataLoader,
        data_len: int,
        ProB: ProcessDABatch,
        DA: torch.nn.modules,
        model: torch.nn.Module,
        optimizer: torch.optim,
        total_loss: Dict[str, torch.tensor]):
    """Function for processing equi and cube outputs"""
    for key, ipt in unlabel.items():
        if key not in ["rgb", "rgb_name"]:
            unlabel[key] = ipt.to(args.device)

    b, c, h, w = unlabel["normalized_rgb"].shape

    # Project CUBE with rotated equi ##################
    pseudo_equi = unlabel["normalized_rgb_noaug"]
    pseudo_equi, rot_vec = ProB.randRotate(pseudo_equi)
    # BiFuse_cube = ProB.BiFuse_E2C.ToCubeTensor(pseudo_equi) # BDFLRU
    BiFuse_cube = ProB.BiFuse_E2C(pseudo_equi) # BDFLRU


    """
    Change from sunset cube to biFuse cube
    """
    cube_inputs = BiFuse_cube
    SIDE_LEN = 518  # Best performance for DA
    cube_inputs = torchvision.transforms.functional.resize(cube_inputs, (SIDE_LEN, SIDE_LEN))

    # Error occurs when input batch size for DA is too large
    with torch.no_grad():
        if cube_inputs.shape[0] > 9:
            center = cube_inputs.shape[0] // 2
            depths = torch.concat([DA(cube_inputs[:center]), DA(cube_inputs[center:])], 0)
        else:
            depths = DA(cube_inputs)
    # 518 -> 256
    depths = F.interpolate(depths.unsqueeze(1), (h//2, h//2), mode='nearest')
    
    # load sky mask for unlabel if key/path exist
    if 'val_mask' in unlabel:
        inputs["pseudo_mask_equi"] = unlabel['val_mask']
        
        # rotate mask based on DA input
        equi_mask_sky = ProB.rotate(inputs['pseudo_mask_equi'].clone().float(), ProB.rot_vec)
        # mask_cube_sky = ProB.BiFuse_E2C.ToCubeTensor(equi_mask_sky).floor() == 1
        mask_cube_sky = ProB.BiFuse_E2C(equi_mask_sky).floor() == 1

    
    # Scale to 0~1
    for idx in range(depths.shape[0]):
        depths[idx] -= depths[idx].min()
        if depths[idx].max() > 0:
            depths[idx] /= depths[idx].max()
    
    ####
    for key, ipt in inputs.items():
        if key not in ["rgb", "rgb_name"]:
            inputs[key] = ipt.to(args.device)
    
    # add pseudo
    if args.need_cube:
        inputs["normalized_cube_rgb"] = torch.concat([inputs["normalized_cube_rgb"], unlabel["normalized_cube_rgb"]], 0)
    inputs["normalized_rgb"] = torch.concat([inputs["normalized_rgb"], unlabel["normalized_rgb"]], 0)
    inputs["rgb"] = torch.concat([inputs["rgb"], unlabel["rgb"]], 0)
    
    """
    Calculate Pseudo on CUBE
    """
    inputs["pseudo_depth"] = depths # B1, D1, F1...
    if 'val_mask' in unlabel:
        inputs["pseudo_mask"] = depths > 0 & mask_cube_sky
    else:
        inputs["pseudo_mask"] = (torch.ones(depths.shape, device=args.device) == 1)


    ####
    equi_inputs = inputs["normalized_rgb"]

    # CUBE INPUT
    if args.need_cube:
        cube_inputs = inputs["normalized_cube_rgb"].to(args.device)
        outputs = model(equi_inputs, cube_inputs)
    else:
        outputs = model(equi_inputs)

    """
    Calculate Pseudo on CUBE
    """
    inputs, outputs, ProB.process_output(inputs, outputs)
    
    if hasattr(args, 'gt_w'):
        losses = model.get_loss(inputs, outputs, float(args.gt_w), float(args.pseudo_w))
    else:
        losses = model.get_loss(inputs, outputs)
    optimizer.zero_grad()
    losses["loss"].backward()
    optimizer.step()
    args.cur_step += 1
    for k, v in losses.items():
        total_loss[f'total_{k}'] += v.data.cpu().numpy() / data_len

    return outputs


def val(
    args: argparse.ArgumentParser.parse_args, 
    model: nn.Module, 
    dataloader: DataLoader, 
    epoch: int, 
    writer: SummaryWriter=None, 
    evaluator: Affine_Inv_Evaluator=None, 
    mode='Valid', 
    save_log_flag: bool=True, 
    save_img_flag: bool=False):
    """Eval function."""
    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader)
        pbar.set_description("{} Epoch_{}".format(mode, epoch))
        total_loss = defaultdict(float)
        for batch_idx, inputs in enumerate(pbar):
            for key, ipt in inputs.items():
                if key not in ["rgb", "rgb_name"]:
                    inputs[key] = ipt.to(args.device)
            equi_inputs = inputs["normalized_rgb"]

            # CUBE INPUT
            if args.need_cube:
                cube_inputs = inputs["normalized_cube_rgb"].to(args.device)
                outputs = model(equi_inputs, cube_inputs)
            else:
                outputs = model(equi_inputs)
            losses = model.get_loss(inputs, outputs)
            # Relative Depth
            if inputs["val_mask"].sum() > 0:
                evaluator.compute_affine_inv_eval_metrics(
                    gt_depth=inputs["gt_metric_depth"].detach(), 
                    pred_depth=outputs["pred_depth"].detach(), 
                    mask=inputs["val_mask"].detach())

            for k, v in losses.items():
                total_loss[f'total_{k}'] += v.data.cpu().numpy() / len(dataloader)

        # Evaluator to shell and tensorboard
        for i, key in enumerate(evaluator.metrics.keys()):
            total_loss[f'total_{key}'] = np.array(evaluator.metrics[key].avg.cpu())
        evaluator.print()
        if save_log_flag:
            save_log(writer, inputs, outputs, total_loss, args)


def main_joint_unlabel():
    # Init
    args, model, optimizer, writer, old_settings = init()
    DA = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format('vitl')).to(args.device).eval()
    DA = torch.nn.DataParallel(DA)

    # Get dataloaders
    loader_dict = get_unlabel_data(args)
    train_loader = loader_dict['train']
    val_loader = loader_dict['val']
    test_loader = loader_dict['test']
    unlabel_loader = loader_dict['unlabel']
    zeroshot_loader = loader_dict['zeroshot']

    # Print State
    print("Training model named:\n ", args.model_name)
    print("Training dataset named:\n ", args.dataset)
    print("Training Exp ID:\n ", args.id)
    print("Models and tensorboard events files are saved to:\n", args.log)
    print("Training is using:\n ", args.device)

    # update epochs and steps
    args.cur_step = 0
    start_epoch = 0
    if old_settings is not None:
        args.cur_step = old_settings.cur_step
        start_epoch = old_settings.cur_epoch
        args.cur_epoch = old_settings.cur_epoch
    evaluator = Affine_Inv_Evaluator(median_align=args.median_align)
    zeroshot_evaluator = Affine_Inv_Evaluator(median_align=args.median_align, crop=68)

    # process depth anything results, either for equi model input or output
    if hasattr(args, 'rot_version'):
        ProB = ProcessDABatch(h=args.h, w=args.w, rot=args.rot_version)
    else:
        ProB = ProcessDABatch(h=args.h, w=args.w)

    # Epoch on Label or Unlabel
    iter_loader = enumerate(unlabel_loader)

    for epoch in tqdm(range(start_epoch, args.epochs), desc='epoch'):
        args.cur_epoch = epoch + 1
        iter_loader = train_joint_unlabel(args, model, DA, iter_loader, train_loader, optimizer, epoch, writer['train'], unlabel_loader, ProB)
        if val_loader is not None:
            evaluator.reset_eval_metrics()
            val(args, model, val_loader, epoch, writer['val'], evaluator)
        if test_loader is not None:
            evaluator.reset_eval_metrics()
            val(args, model, test_loader, epoch, writer['test'], evaluator, mode='Test')
        if zeroshot_loader is not None:
            zeroshot_evaluator.reset_eval_metrics()
            val(args, model, zeroshot_loader, epoch, writer['zeroshot'], zeroshot_evaluator, mode='zeroshot')
        
        if (epoch + 1) % args.save_every == 0:
            save_model(model, optimizer, args)
    if test_loader is not None:
        evaluator.reset_eval_metrics()
        val(args, model, test_loader, max(args.cur_epoch, args.epochs), writer['test'], evaluator, mode='Test')
    
    if zeroshot_loader is not None:
        zeroshot_evaluator.reset_eval_metrics()
        val(args, model, zeroshot_loader, max(args.cur_epoch, args.epochs), writer['zeroshot'], zeroshot_evaluator, mode='Zeroshot')


if __name__ == '__main__':
    main_joint_unlabel()
