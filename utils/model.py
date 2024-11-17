import os
import sys

import torch

from baseline_models.UniFuse.networks import UniFuse
from baseline_models import BiFuseV2
sys.path.append("baseline_models/HoHoNet/")
from baseline_models.HoHoNet.lib.model.hohonet import HoHoNet
sys.path.append("baseline_models/EGformer/")
from baseline_models.EGformer.models.egformer import EGDepthModel
from utils.metric import ScaleAndShiftInvariantLoss


def get_model(model_name, device, pretrained_dict=None, model_dict=None):
    if model_name.upper() == 'UNIFUSE':
        model = UniFuse(**model_dict)
    elif model_name.upper() == 'BIFUSEV2':
        # BiFuseV2
        dnet_args = {
            'layers': model_dict['layers'],
            'CE_equi_h': model_dict['ce_equi_h']
        }
        if 'sigmoid' in model_dict:
            print("USE Sigmoid relative depth")
            model = BiFuseV2.BiFuse.SupervisedCombinedModelDict(model_dict['save_path'], dnet_args, sigmoid=True)
        else:
            model = BiFuseV2.BiFuse.SupervisedCombinedModelDict(model_dict['save_path'], dnet_args)
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
    elif model_name.upper() == 'EGFORMER':
        model = EGDepthModel(hybrid=False)
    else:
        raise NotImplementedError(f"{model_name} not implemented")
    model.to(device)

    if pretrained_dict:
        model_dict = model.state_dict()
        # some model trained with nn.DataParallel
        pretrained_keys = [k for k in pretrained_dict.keys() if k.startswith('module.')]
        if len(pretrained_keys) > 0:
            # trained with DataParallel, remove module from pretrained dict
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        else:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # load
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.get_loss = ScaleAndShiftInvariantLoss()

    return model


def save_model(model, optim, args):
    """Save model to .pth file."""
    save_path = os.path.join(args.save_folder, f"ckpt_{args.cur_epoch}.pth")
    to_save = dict()
    if hasattr(model, 'module'):
        # nn.DataParallel
        to_save['model'] = model.module.state_dict()
    else:
        to_save['model'] = model.state_dict()

    to_save['optim'] = optim.state_dict()
    to_save['settings'] = args
    torch.save(to_save, save_path)