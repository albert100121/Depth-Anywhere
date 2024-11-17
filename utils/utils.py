import os
import pickle

import torch
import torch.nn as nn
import numpy as np


def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" ")[:2])
    return rgb_depth_list


def read_list_with_ndarray(list_file):
    # rgb_depth_list = np.empty((0, 2))
    rgb_list = np.empty(0)
    depth_list = np.empty(0)
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            # rgb_depth_list = np.append(rgb_depth_list, [line.strip().split(" ")], 0)
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
    # for m in module.modules():
    #     if isinstance(m, nn.Linear):
    #         group_decay.append(m.weight)
    #         if m.bias is not None:
    #             group_no_decay.append(m.bias)
    #     elif isinstance(m, nn.modules.conv._ConvNd):
    #         group_decay.append(m.weight)
    #         if m.bias is not None:
    #             group_no_decay.append(m.bias)
    #     elif isinstance(m, nn.modules.batchnorm._BatchNorm):
    #         if m.weight is not None:
    #             group_no_decay.append(m.weight)
    #         if m.bias is not None:
    #             group_no_decay.append(m.bias)
    #     elif isinstance(m, nn.GroupNorm):
    #         if m.weight is not None:
    #             group_no_decay.append(m.weight)
    #         if m.bias is not None:
    #             group_no_decay.append(m.bias)
    #     elif isinstance(m, nn.LayerNorm):
    #         if m.weight is not None:
    #             group_no_decay.append(m.weight)
    #         if m.bias is not None:
    #             group_no_decay.append(m.bias)
    #     else:
    #         import pdb
    #         # if m.bias is not None:
    #         #     group_no_decay.append(m.bias)

    for name, param in module.named_parameters():
        if name.endswith('weight'):
            group_decay.append(param)
        elif name.endswith('bias'):
            group_no_decay.append(param)
    # assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    return [
        dict(params=[param for param in group_decay if param.requires_grad]),
        dict(params=[param for param in group_no_decay if param.requires_grad], weight_decay=.0)
    ]


# Some recommendations for memory leak (not solution for this time)
# https://github.com/Lightning-AI/pytorch-lightning/issues/17257
# https://github.com/ppwwyyxx/RAM-multiprocess-dataloader/blob/79897b26a2c4185a3ed086f18be5ea300913d5b7/serialize.py#L40-L50
class NumpySerializedList():
    def __init__(self, lst: list):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        print(
            "Serializing {} elements to byte tensors and concatenating them all ...".format(
                len(lst)
            )
        )
        self._lst = [_serialize(x) for x in lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = np.cumsum(self._addr)
        self._lst = np.concatenate(self._lst)
        print("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr])
        return pickle.loads(bytes)


class TorchSerializedList(NumpySerializedList):
    def __init__(self, lst: list):
        super().__init__(lst)
        self._addr = torch.from_numpy(self._addr)
        self._lst = torch.from_numpy(self._lst)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        return pickle.loads(bytes)
