import argparse
import configparser


def force_config_value_type(val):
    if val.isdecimal():
        return int(val)
    else:
        try:
            return float(val)
        except:
            if val.upper() == 'TRUE':
                return True
            elif val.upper() == 'FALSE':
                return False
            else:
                return val


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config')

    # Read experiment setting from config file first
    config_args, remaining_argv = parser.parse_known_args()
    config = configparser.ConfigParser()
    config.read(config_args.config)
    default = dict(config['EXP_SETTING'])
    for k, v in default.items():
        if ',' in v:
            default[k] = list(map(float, v.split(',')))
        if type(v) == str and 'true' in v.lower():
            default[k] = True
        elif type(v) == str and 'false' in v.lower():
            default[k] = False

    # Placeholder setting from config
    parser.add_argument('--id',
                        help='experiment id to name checkpoints and logs')

    """
    Dataset & Augmentation setting
    """
    parser.add_argument('--root',
                        help='root directory to training dataset. '
                        'should contains img, label_cor subdirectories')
    parser.add_argument('--train_txt')
    parser.add_argument('--val_txt')
    parser.add_argument('--test_txt')
    parser.add_argument('--unlabel_root',
                        help='root directory to training dataset. '
                        'should contains img, label_cor subdirectories')
    parser.add_argument('--unlabel_train_txt')
    parser.add_argument('--zero_shot_root',
                        help='root directory to training dataset. '
                        'should contains img, label_cor subdirectories')
    parser.add_argument('--zero_shot_txt')
    parser.add_argument('--dataset', help='the dataset of this training')
    parser.add_argument('--epochs', type=int, help='epochs to train')
    parser.add_argument('--batch_size_train',
                        type=int,
                        help='training mini-batch size')
    parser.add_argument('--batch_size_unlabel',
                        type=int,
                        help='unlabel data mini-batch size')
    parser.add_argument('--batch_size_val',
                        type=int,
                        help='validation mini-batch size')
    parser.add_argument('--num_workers',
                        type=int,
                        help='numbers of workers for dataloaders')
    parser.add_argument('--model_name', default='PanoFormer', choices=['PanoFormer', 'BiFuse'])
    parser.add_argument('--relative', default=None, help='SSI Loss v1 or v2')
    parser.add_argument('--median_align', action='store_true', help='Apply median align in metric calculation')
    parser.add_argument('--no_vflip',
                        action='store_true',
                        help='stop top-bottom flip augmentation')
    parser.add_argument('--no_hflip',
                        action='store_true',
                        help='stop left-right flip augmentation')
    parser.add_argument('--disable_color_augmentation',
                        action='store_true',
                        help='stop color augmentation')
    parser.add_argument('--disable_LR_filp_augmentation',
                        action='store_true',
                        help='stop left-right flip augmentation')
    parser.add_argument('--disable_yaw_rotation_augmentation',
                        action='store_true',
                        help='stop yaw direction rotation')
    parser.add_argument('--brightness', type=float, default=1.0, help='brightness scaling factor')
    parser.add_argument('--contrast', type=float, default=1.0, help='contrast scaling factor')
    parser.add_argument('--saturation', type=float, default=1.0, help='saturation')
    parser.add_argument('--hue', default=1.0, type=float, help='hue')
    parser.add_argument('--need_cube',
                        action='store_true',
                        help='use cube in model input')
    """
    Pre-processing setting
    """
    parser.add_argument('--h', default=512, type=int, help='loader process height')
    parser.add_argument('--w', default=1024, type=int, help='loader process width')
    parser.add_argument('--rgb_mean', default=[0.485, 0.456, 0.406], nargs=3, type=float)
    parser.add_argument('--rgb_std', default=[0.229, 0.224, 0.225], nargs=3, type=float)

    """
    Optimizer setting
    """
    parser.add_argument('--optim', help='optimizer to use')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)

    """
    Misc setting
    """
    parser.add_argument('--pth', help='pretrained model')
    parser.add_argument('--seed', type=int, help='manual seed')
    parser.add_argument('--save_every',
                        type=int,
                        help='epochs frequency to save state_dict')
    parser.add_argument('--save_img_every',
                        type=int,
                        help='epochs frequency to save state_dict')
    parser.add_argument('--run_val_every',
                        type=int,
                        default=1,
                        help='epochs frequency to run validation')
    parser.add_argument('--ckpt', help='folder to output checkpoints')
    parser.add_argument('--log', help='folder to logging')
    parser.add_argument('--no_cuda', help='disable cuda', type=int)
    parser.set_defaults(**default)

    # Read from remaining command line setting (replace setting in config)
    args = parser.parse_args(remaining_argv)

    # Parse model setting
    args_model = {
        'model_setting': dict(config['model_setting']),
        'model_kwargs': dict(config['model_kwargs']),
    }
    for key in ['model_setting', 'model_kwargs']:
        for k, v in args_model[key].items():
            if ',' in v:
                args_model[key][k] = list(map(force_config_value_type, v.strip(',').split(',')))
            else:
                args_model[key][k] = force_config_value_type(v)


    return args, args_model


if __name__ == '__main__':
    print(parse_args())