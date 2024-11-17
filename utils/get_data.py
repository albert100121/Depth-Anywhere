from torch.utils.data import DataLoader

from utils.Datasets import UnlabelDataset, LabelDataset, Stanford2D3D


def get_unlabel_data(args):
    """Get dataloader for label/unlabel/zeroshot."""
    if not hasattr(args, 'batch_size_unlabel'):
        args.batch_size_unlabel = args.batch_size_train // 2
        args.batch_size_train /= 2

    train_loader = unlabel_loader = val_loader = test_loader = zero_shot_loader = None
    # TRAIN
    train_dataset = LabelDataset(
        root_dir=args.root, 
        list_file=args.train_txt, 
        height=args.h, 
        width=args.w,
        disable_color_augmentation=args.disable_color_augmentation,
        disable_LR_filp_augmentation=args.disable_LR_filp_augmentation,
        disable_yaw_rotation_augmentation=args.disable_yaw_rotation_augmentation,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        hue=args.hue,
        mean=args.rgb_mean,
        std=args.rgb_std,
        is_training=True,
        relative=args.relative,
        device=args.device,
        need_cube=args.need_cube)

    train_loader = DataLoader(train_dataset, args.batch_size_train, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # UNLABEL
    unlabel_dataset = UnlabelDataset(
        root_dir=args.unlabel_root, 
        list_file=args.unlabel_train_txt, 
        height=args.h, 
        width=args.w,
        disable_color_augmentation=args.disable_color_augmentation,
        disable_LR_filp_augmentation=args.disable_LR_filp_augmentation,
        disable_yaw_rotation_augmentation=args.disable_yaw_rotation_augmentation,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        hue=args.hue,
        mean=args.rgb_mean,
        std=args.rgb_std,
        is_training=True,
        relative=args.relative,
        device=args.device,
        need_cube=args.need_cube)
    unlabel_loader = DataLoader(unlabel_dataset, args.batch_size_unlabel, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # VALID
    if args.val_txt:
        val_dataset = LabelDataset(
            root_dir=args.root, 
            list_file=args.val_txt, 
            height=args.h, 
            width=args.w, 
            disable_color_augmentation=True,
            disable_LR_filp_augmentation=True,
            disable_yaw_rotation_augmentation=True,
            mean=args.rgb_mean,
            std=args.rgb_std,
            is_training=False,
            relative=args.relative,
            device=args.device,
            need_cube=args.need_cube)
        val_loader = DataLoader(val_dataset, args.batch_size_val, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # TEST
    if args.test_txt:
        test_dataset = LabelDataset(
            root_dir=args.root, 
            list_file=args.test_txt, 
            height=args.h, 
            width=args.w, 
            disable_color_augmentation=True,
            disable_LR_filp_augmentation=True,
            disable_yaw_rotation_augmentation=True,
            mean=args.rgb_mean,
            std=args.rgb_std,
            is_training=False,
            relative=args.relative,
            device=args.device,
            need_cube=args.need_cube)
        test_loader = DataLoader(test_dataset, args.batch_size_val, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=False)
    if args.zero_shot_txt:
        zero_shot_dataset = Stanford2D3D(
            root_dir=args.zero_shot_root, 
            list_file=args.zero_shot_txt, 
            height=args.h, 
            width=args.w, 
            disable_color_augmentation=True,
            disable_LR_filp_augmentation=True,
            disable_yaw_rotation_augmentation=True,
            mean=args.rgb_mean,
            std=args.rgb_std,
            is_training=False,
            relative=args.relative,
            device=args.device,
            need_cube=args.need_cube)
        zero_shot_loader = DataLoader(zero_shot_dataset, args.batch_size_val, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=False)

    loader_dict = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'unlabel': unlabel_loader,
        'zeroshot': zero_shot_loader
    }

    return loader_dict
