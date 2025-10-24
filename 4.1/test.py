import argparse
from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import models, transforms

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import course datasets (adjust names if they differ in your `datasets.py`)
from datasets import FrameVideoDataset, FlowVideoDataset  # noqa: F401  # type: ignore


DATA_ROOT="/dtu/datasets1/02516/ucf101_noleakage"

IMAGE_SIZE = 112

transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE)),
        transforms.ToTensor()
    ])

# train_loader = DataLoader(ds_train, batch_size=8, shuffle=True,
#                                   num_workers=5, pin_memory=True)

# batch = next(iter(train_loader))
# flows, labels = batch
# print("Batch flow shape:", flows.shape)
# print("Batch labels shape:", labels.shape)




ds_train = FrameVideoDataset(root_dir=DATA_ROOT, split="train",
                                 transform=transform, stack_frames=False)

frame_sample, label = ds_train[0]
print(f"Number of frames: {len(frame_sample)}")
print(f"Shape frame: {frame_sample.shape}")
print("Label:", label)

train_loader = DataLoader(ds_train, batch_size=64, shuffle=True,
                                  num_workers=5, pin_memory=True)

batch = next(iter(train_loader))
frames, labels = batch
print(f"frames shape: {frames.shape}")
print(f"Shape of one frame: {frames[0].shape}")
print("Label:", labels.shape)


# ds_flow_train = FlowVideoDataset(root_dir=DATA_ROOT, split="train",
#                                  transform=transform)

# flow_sample, label = ds_flow_train[0]

# print("Flow sample shape:", flow_sample.shape)
# print("Label:", label)


# class TwoStreamDataset(torch.utils.data.Dataset):
#     def __init__(self, rgb_dataset, flow_dataset):
#         assert len(rgb_dataset) == len(flow_dataset), "Datasets must be same length"
#         self.rgb_dataset = rgb_dataset
#         self.flow_dataset = flow_dataset

#     def __len__(self):
#         return len(self.rgb_dataset)

#     def __getitem__(self, idx):
#         rgb, label_rgb = self.rgb_dataset[idx]
#         flow, label_flow = self.flow_dataset[idx]
#         assert label_rgb == label_flow, "Labels must match"
#         return rgb, flow, label_rgb

# ds = TwoStreamDataset(ds_train, ds_flow_train)


# frame_sample1, flow_sample1, label1 = ds[0]



# print(f"Number of frames: {len(frame_sample1)}")
# print(f"Shape of one frame: {frame_sample1[0].shape}")
# print("Label:", label1)

# print("Flow sample shape:", flow_sample1.shape)
# print("Label:", label1)
