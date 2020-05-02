"""Visualize feature distribution."""
import os
import ipdb
import h5py

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import parse_rec

label_name = {0: 'None', 
1: 'attach leg 1', 
2: 'attach leg 2', 
3: 'attach leg 3', 
4: 'attach leg 4', 
5: 'detach leg 1', 
6: 'detach leg 2', 
7: 'detach leg 3', 
8: 'detach leg 4', 
9: 'flip over', 
10: 'spin in', 
11: 'spin out', 
12: 'pick leg'}

phase = 'flow'
pose_flag = False

root_dir = '../dataset_public/ikea-fa-release-data'
rec_file = 'test_30fps.csv'
feat_file = 'data/feats_len5.h5'
recs = parse_rec(root_dir, rec_file)

vid_middle = recs[0][:-4]
feat_path = os.path.join(root_dir, feat_file)
f = h5py.File(feat_path, 'r')

# read feature data according to phases
if phase == 'rgb':
    rgb_feats = torch.tensor(f[vid_middle + '/rgb_feats']).squeeze()
    feats = rgb_feats
elif phase == 'flow':
    flow_feats = torch.tensor(f[vid_middle + '/flow_feats']).squeeze()
    feats = flow_feats
elif phase == 'rgb_flow':
    rgb_feats = torch.tensor(f[vid_middle + '/rgb_feats']).squeeze()
    flow_feats = torch.tensor(f[vid_middle + '/flow_feats']).squeeze()
    feats = torch.cat((rgb_feats, flow_feats), 1)
elif phase == 'pose':
    pose_feats = torch.tensor(f[vid_middle + '/pose_feats']).squeeze()
    feats = pose_feats
    pose_flag = False
else:
    print("Only support phase = 'rgb', 'flow' or 'pose' ")

# whether to add additional feature
if pose_flag:
    pose_feats = torch.tensor(f[vid_middle + '/pose_feats']).squeeze()
    feats = torch.cat((feats, pose_feats), 1)

# original labels
labels = torch.tensor(f[vid_middle + '/labels'])
f.close()

writer = SummaryWriter('log_test')
writer.add_embedding(feats, metadata=labels.numpy().tolist())
writer.close()

