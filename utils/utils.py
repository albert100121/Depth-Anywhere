import numpy as np


def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" ")[:2])
    return rgb_depth_list


def read_list_with_ndarray(list_file):
    rgb_list = np.empty(0)
    depth_list = np.empty(0)
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb, depth = line.strip().split(" ")
            rgb_list = np.append(rgb_list, rgb)
            depth_list = np.append(depth_list, depth)
    rgb_list = np.expand_dims(rgb_list, 1)
    depth_list = np.expand_dims(depth_list, 1)
    rgb_depth_list = np.concatenate([rgb_list, depth_list], 1)

    return rgb_depth_list


def group_weight(module):
    group_decay = []
    group_no_decay = []

    for name, param in module.named_parameters():
        if name.endswith('weight'):
            group_decay.append(param)
        elif name.endswith('bias'):
            group_no_decay.append(param)
    return [
        dict(params=[param for param in group_decay if param.requires_grad]),
        dict(params=[param for param in group_no_decay if param.requires_grad], weight_decay=.0)
    ]
