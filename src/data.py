'''Prepare dataloaders for model input.
'''
import torch.utils.data as data
import torch

import os, sys, random, cv2, ipdb
import csv, json, h5py
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool


class DataPrefetcher():
    '''Use data_prefether to open a new cuda stream to copy tensor to gpu.
    '''
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_input = self.next_input
        next_target = self.next_target
        self.preload()
        return next_input, next_target


# ------------------------------------------------------------------------------
# Read features rather than original data from saved h5 files.
# Still read one sequence once, but the dimensions of features are much lower 
# than those of original data. So it is much faster than the fourth try.
def load_feat(f, seq_idx, mode_name):
    data_path = os.path.join(str(seq_idx).zfill(2), mode_name)
    seq_data = torch.tensor(f[data_path])
    return seq_data


def load_feats(feat_path, mode, pose_flag, pose_1080p_flag=True, train_flag=True):
    """ For every sequence, loading all feats from saved h5 file to a list.
    It supports 'rgb', 'flow', 'pose' and 'label', which share samilar data structures.
    """
    # training with the first 30 sequences: (00, 29), 32
    # testing with the following 20 sequences: (33, 44), 46,  (50, 55), (56, 59)
    # if train_flag:
    #     ranges = range(0, 30)
    # else:
    #     ranges = range(30, 50)
    
    # filter out thoses sequences with bad poses
    # 2020-02-13
    # if train_flag:
    #     ranges = [i for i in range(0, 29)]
    #     ranges.append(32)
    # else:
    #     ranges = [i for i in range(33, 44)] + [i for i in range(50, 55)] \
    #             + [i for i in range(56, 59)]
    #     ranges.append(46)
    #     # take two sequences fro visualization
    #     # ranges = [33, 39]
    
    # The first 41 for training and the rest 9 for testing
    if train_flag:
        ranges = [i for i in range(0, 29)] + [i for i in range(32, 44)]
    else:
        ranges = [i for i in range(50, 55)] + [i for i in range(56, 59)]
        ranges.append(46)
        # take two sequences fro visualization
        # ranges = [46, 50]
    # load all the sequences and labels into a list
    feats = []
    poses = []
    labels = []
    
    f = h5py.File(feat_path, 'r', libver='latest', swmr=True)
    for seq_idx in ranges:

        # load feats
        print('Loading seq_feats #', str(seq_idx).zfill(2))
        seq_rgb = load_feat(f, seq_idx, 'rgb')
        seq_flow = load_feat(f, seq_idx, 'flow')
        seq_label = load_feat(f, seq_idx, 'label')
        if pose_1080p_flag:
            seq_pose = load_feat(f, seq_idx, 'pose_1080p')
        else:
            seq_pose = load_feat(f, seq_idx, 'pose')
        
        # config feats
        if mode == 'rgb':
            seq_feat = seq_rgb 
        elif mode == 'flow':
            seq_feat = seq_flow
        elif mode == 'pose':
            seq_feat = seq_pose
        elif mode == 'rgb_flow':
            seq_feat = torch.cat((seq_rgb, seq_flow), 1)[:]
        else:
            print("Only support mode = 'rgb', 'flow', 'pose' or 'rgb_flow'")
        
        if pose_flag:
            seq_feat = torch.cat((seq_feat, seq_pose), 1)[:]

        # put all feats in to one list
        feats.append(seq_feat)
        labels.append(seq_label)
        poses.append(seq_pose)
    
    f.close()
    return feats, labels, poses


class OAD_feat_seq2seq(data.Dataset):
    '''Read features from saved h5 files. Used in triain_lstm_multi_label.py 
    Almost the same with Ikea_feat_cls_reg. Bur it returns the labels and poses 
    of next future sequences for the joint seq2seq model. 
    '''
    def __init__(self, feat_path, mode, future_length, num_joint, pose_flag, 
                pose_1080p_flag, train_flag):
        
        self.mode = mode
        self.future_length = future_length
        self.num_joint = num_joint
        self.feats, self.labels, self.poses = load_feats(feat_path, mode, 
                                    pose_flag, pose_1080p_flag, train_flag)
        # ipdb.set_trace()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq_feats = self.feats[idx]
        # whether to use future feature
        if self.future_length > 0:
            new_length = seq_feats.size(0) - self.future_length
            seq_feats = seq_feats[:new_length]
            seq_labels = torch.empty(self.future_length, new_length)
            dim_pose = self.num_joint * 2
            seq_poses = torch.empty(self.future_length, new_length, dim_pose)
            
            for k in range(self.future_length):
                # ipdb.set_trace()
                seq_labels[k] = self.labels[idx][k : new_length+k][:]
                seq_poses[k] = self.poses[idx][k: new_length+k, -dim_pose:][:]

        return seq_feats, seq_labels, seq_poses


# ------------------------------------------------------------------------------
# Fourth try:
# OAD dataset is relatively small (8.5GB for rgbs.h5 and 19.1GB for flows.h5). 
# So it is feasible to load all the dataset into a list for fastest iteration. 
# I recommend not loading them into a dict as it will take much more memory than
# a list.Though all the sequences are stored in one h5 file, loading them to memory 
# still needs to perform one by one. Maybe you can try to save all sequences to 
# one varible then save it into h5 file. I did not do this because the lengths of 
# these sequences are different. It will take more spaces if forcing to save them
# to one numpy array with zero paddings. Also, I do not believe it will improve
# a lot of doing so.

def norm_seq(data, mean, std, num_channel=3):
    nframes = data.shape[0]
    out = np.empty((num_channel * nframes,
                    data.shape[1], data.shape[2]), np.float32)

    # np.float32 for fast reshape
    data = data.astype('float32')
    # whether to divide 255
    if num_channel == 3:    # rgb
        max_value = 255
    else:                   # flow
        max_value = 1

    # transpose by frames
    for kk in range(nframes):
        # Reshape (H x W x C) to (C x H x W)
        out[num_channel*kk : num_channel*(kk+1), :, :] = (
            data[kk].transpose(2, 0, 1)) / max_value
    
    # normalize by chabnnels
    for i in range(num_channel):
        out[i::num_channel] -= mean[i]
        out[i::num_channel] /= std[i]
    
    return out

def reshape_pose(pose):
    # ipdb.set_trace()
    frame_length, num_joint, num_channel = pose.shape   # 5, 15, 2
    dim_pose = num_channel * num_joint                  # 2x15=30
    dim_clip = dim_pose * frame_length                  # 30x5=150
    clip_data = np.zeros((dim_clip), dtype=np.float32)
    for kk in range(frame_length):
        clip_data[dim_pose*kk : dim_pose*(kk+1)] = pose[kk].reshape(dim_pose)[:]
    return clip_data

def load_seq(file_path, mode, seq_idx):
    '''Load one sequence at a time for multi dataloaders.
    '''
    f = h5py.File(file_path, 'r', libver='latest', swmr=True)
    data_path = os.path.join(str(seq_idx), mode)
    label_path = os.path.join(str(seq_idx), 'label')
    print('Loading #', data_path)
    
    # speed up with read_direct()
    data = f[data_path]
    seq_data = np.empty(data.shape, data.dtype) 
    data.read_direct(seq_data)
    # read labes
    seq_label = np.array(f[label_path], dtype=data.dtype)

    # the bumber of flow frames is 1 less than that of rgb/pose frames
    if mode != 'flow':
        seq_data = seq_data[1:]
        seq_label = seq_label[1:]

    f.close()
    return seq_data, seq_label
    
def load_dataset(file_path, mode, train_flag=True):
    """ For every sequence, loading all sequences from saved h5 file to a list.
    It supports 'rgb', 'flow', 'pose' and 'label', which share samilar data structures.
    """
    # training with the first 30 sequences: [00, 29]
    # testing with the following 20 sequences: [30, 49]
    if train_flag:
        ranges = range(0, 50)
    else:
        ranges = range(30, 50)

    # load all the sequences and labels into a list
    dataset = []
    labels = []
    for seq_idx in ranges:
        seq_data, seq_label = load_seq(file_path, mode, seq_idx)
        dataset.append(seq_data)
        labels.append(seq_label)
    
    return dataset, labels


class OAD_single(data.Dataset):
    '''Prepare for the inputs of single stream: rgb, flow, or gray.
    Read rgb/flow from the h5 files of each video. Used in train_fc_oad.py
    This class takes the pre-loaded whole dataset and labels as input.
    '''
    def __init__(self, root, seq_data, seq_label, mode, frame_length):
         # read feature data according to phases
        if mode == 'rgb':
            self.num_channel = 3
            self.mean = [0.485, 0.456, 0.406] * frame_length
            self.std = [0.229, 0.224, 0.225] * frame_length
        elif mode == 'flow':
            self.num_channel = 2
            self.mean = [0.5, 0.5] * frame_length
            self.std = [0.226, 0.226] * frame_length
        elif mode == 'pose':
            self.num_channel = 2
        else:
            print("Only support mode = 'rgb', 'flow' or 'pose' ")
            
        # Change to read all data before training.
        self.seq_data = seq_data
        self.seq_label = seq_label
        self.frame_length = frame_length
        self.mode = mode

    def __len__(self):
        return len(self.seq_label) - self.frame_length

    def __getitem__(self, idx):
        # current frame time: new_idx-1. 
        new_idx = idx + self.frame_length
        # example: data.shape = (5, 224, 224, 3)
        data = self.seq_data[idx:new_idx]
        # example: clip_data.shape = (224, 224, 15)
        # ipdb.set_trace()
        clip_data = norm_seq(data, self.mean, self.std, self.num_channel)
        clip_data = torch.from_numpy(clip_data)
        # next frame time: new_idx
        label = torch.tensor( self.seq_label[new_idx], dtype=torch.long)

        return clip_data, label


class OAD_single_test(data.Dataset):
    '''Prepare for the inputs of single stream: rgb, flow, or gray.
    Read rgb/flow from the h5 files of each video. Used in test_fc_one_stream.py
    This class loads data and labels at every dataloader rather than pre-loading 
    the whole dataset before testing.
    '''
    def __init__(self, root, seq_idx, mode, frame_length):
         # read feature data according to phases
        if mode == 'rgb':
            self.num_channel = 3
            data_name = 'rgbs.h5'
            self.mean = [0.485, 0.456, 0.406] * frame_length
            self.std = [0.229, 0.224, 0.225] * frame_length
        elif mode == 'flow':
            self.num_channel = 2
            data_name = 'flows.h5'
            self.mean = [0.5, 0.5] * frame_length
            self.std = [0.226, 0.226] * frame_length
        elif mode == 'pose':
            self.num_channel = 2
            self.num_joint = 15
            data_name = 'poses.h5'
        elif mode == 'pose_1080p':
            self.num_channel = 2
            self.num_joint = 15
            data_name = 'poses_1080p.h5'
        else:
            print("Only support mode = 'rgb', 'flow' or 'pose' ")
            
        file_path = os.path.join(root, 'group_data', data_name)
        self.seq_data, self.seq_label = load_seq(file_path, mode, seq_idx)
        self.frame_length = frame_length
        self.mode = mode
        
    def __len__(self):
        return len(self.seq_label) - self.frame_length

    def __getitem__(self, idx):
        # current frame time: new_idx-1. 
        new_idx = idx + self.frame_length
        data = self.seq_data[idx:new_idx]
        
        # ipdb.set_trace()
        if self.mode in ['pose', 'pose_1080p']:
            # reshape pose 
            # example: data.shape = (5, 15, 2)
            # clip_data.shape = (10, 15)
            clip_data = reshape_pose(data)
        else:   
            # normalize rgb / flow
            # example: data.shape = (5, 224, 224, 3)
            # clip_data.shape = (15, 224, 224)
            clip_data = norm_seq(data, self.mean, self.std, self.num_channel)
        clip_data = torch.from_numpy(clip_data)
        # next frame time: new_idx
        label = torch.tensor( self.seq_label[new_idx], dtype=torch.long)

        return clip_data, label


# ------------------------------------------------------------------------------
# Third try:
# Read features rather than original data from saved h5 files.
# Still read one sequence once, but the dimensions of features are much lower 
# than those of original data. So it is much faster than the second try.

class Ikea_feat_seq2seq(data.Dataset):
    '''Read features from saved h5 files. Used in triain_lstm_multi_label.py 
    Almost the same with Ikea_feat_cls_reg. Bur it returns the labels and poses 
    of next future sequences for the joint seq2seq model. 
    '''
    def __init__(self, root, rec, phase, pose_flag, feat_file, future_length):
        # modality: 'frames', 'flows'...
        self.rec = rec
        self.phase = phase
        self.pose_flag = pose_flag
        self.feat_path = os.path.join(root, feat_file)
        self.future_length = future_length
        # ipdb.set_trace()

    def __len__(self):
        return len(self.rec)

    def __getitem__(self, idx):
        feats, labels, poses = read_feat(idx, self.rec, self.phase, \
            self.pose_flag, self.feat_path)

        # whether to use future feature
        if self.future_length > 0:
            new_length = feats.size(0) - self.future_length
            feats = feats[:new_length]
            seq_labels = torch.zeros(self.future_length, new_length)
            seq_poses = torch.zeros(self.future_length, new_length, 16)
            
            for k in range(self.future_length):
                # ipdb.set_trace()
                seq_labels[k] = labels[k : new_length+k][:]
                seq_poses[k] = poses[k: new_length+k, -16:][:]

        return feats, seq_labels, seq_poses


class Ikea_feat_cls_reg(data.Dataset):
    '''Read features from saved h5 files. Used in triain_lstm_multi_label.py 
    Almost the same with Ikea_feat. Bur it returns the last 1 (future_length) 
    label and pose for the joint classification and regression model. 
    '''
    def __init__(self, root, rec, phase, pose_flag, feat_file, future_length):
        # modality: 'frames', 'flows'...
        self.rec = rec
        self.phase = phase
        self.pose_flag = pose_flag
        self.feat_path = os.path.join(root, feat_file)
        self.future_length = future_length
        # ipdb.set_trace()

    def __len__(self):
        return len(self.rec)

    def __getitem__(self, idx):
        feats, labels, poses = read_feat(idx, self.rec, self.phase, \
            self.pose_flag, self.feat_path)

        # whether to use future feature
        if self.future_length > 0:
            # new_length = feats.size(0) - self.future_length
            # labels = labels[:new_length]          # next one label
            labels = labels[self.future_length:]    # next future_length label
            poses = poses[self.future_length:, -16:]# The last 16 stands for one pose
            feats = feats[self.future_length:]
        return feats, labels, poses


class Ikea_feat_multi_label(data.Dataset):
    '''Read features from saved h5 files. Used in triain_lstm_multi_label.py 
    Almost the same with Ikea_feat. Bur it returns multi labels for the multi-label 
    classification model. 
    '''
    def __init__(self, root, rec, phase, pose_flag, feat_file, future_length):
        # modality: 'frames', 'flows'...
        self.rec = rec
        self.phase = phase
        self.pose_flag = pose_flag
        self.feat_path = os.path.join(root, feat_file)
        self.future_length = future_length
        # ipdb.set_trace()

    def __len__(self):
        return len(self.rec)

    def __getitem__(self, idx):
        feats, labels, pose_feats = read_feat(idx, self.rec, self.phase, \
            self.pose_flag, self.feat_path)

        # whether to use future feature
        if self.future_length > 0:
            new_length = feats.size(0) - self.future_length
            feats = feats[:new_length]
            multi_labels = torch.zeros(self.future_length, new_length)
            for k in range(self.future_length):
                multi_labels[k] = labels[k : new_length+k][:]
        return feats, multi_labels


class Ikea_feat(data.Dataset):
    '''Read features from saved h5 files. Used in train_lstm.py and train_gru.py
    
    The feature is extracted by a pretrained classification resnet-18. The feature 
    is the feature map after the last avgpool (before dropout), which dimension 
    is 512 for resnet-18 and resnet-34 (2048 for resnet-50 or higher).
    '''
    def __init__(self, root, rec, phase, pose_flag, feat_file, future_length):
        # modality: 'frames', 'flows'...
        self.rec = rec
        self.phase = phase
        self.pose_flag = pose_flag
        self.feat_path = os.path.join(root, feat_file)
        self.future_length = future_length
        # ipdb.set_trace()

    def __len__(self):
        return len(self.rec)

    def __getitem__(self, idx):
        feats, labels, pose_feats = read_feat(idx, self.rec, self.phase, \
            self.pose_flag, self.feat_path)

        # whether to use future feature
        if self.future_length > 0:
            new_length = feats.size(0) - self.future_length
            labels = labels[self.future_length:]    # next future_length label
            new_feats = torch.cat((feats, feats), 1)
            
            for k in range(new_length):
                new_feats[k] = torch.cat((feats[k][:], feats[k+self.future_length]), 0)
            feats = new_feats[:new_length]
            # ipdb.set_trace()
        return feats, labels


def read_feat(idx, rec, phase, pose_flag, feat_path):
    """ For every sequence, loading features and labels from saved h5 file to memory.
    """
    vid_middle = rec[idx][:-4]
    f = h5py.File(feat_path, 'r')

    # read feature data according to phases
    pose_feats = 0
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

    return feats, labels, pose_feats




# ------------------------------------------------------------------------------
# Second try:
# Read one video sequence at every dataloader rather than one frame at a dataloader.
# We try to speed up loading data using multi dataloaders as well as storing a 
# sequence into one h5 file. Specifically, we prepare a dataloader for every video
# to avoid loading frames one by one. We though about storing all data into one 
# h5 file, however, we have such limited memorys (maximum 64GB) that we cannot 
# load all the data at once.

class Ikea_seq(data.Dataset):
    '''Prepare single stream input with pose. Used in train_lstm_online_feat and 
    old versions of train_lstm.py and train_gru.py. 
    Note that the pose is extraced from original 1080P images. Therefore, it needs 
    to be normolized to [0, 1]. Remember to normalize back to your preferred size
    (e.g., 360P) when visualizing it.
    Also, as the authors emphasize in the paper, only the first 8 joints (upper 
    body) are reliable because the lower is blocked by the bench in most cases.
    '''
    def __init__(self, root, rec, phase, frame_length, transform):
        # modality: 'frames', 'flows'...
        if phase == 'rgb':
            modality = 'frames'
            self.num_channel = 3
        elif phase == 'flow':
            modality = 'flows'
            self.num_channel = 2
        elif phase == 'gray':
            modality = 'frames'
            self.num_channel = 1
        else:
            print("Only support phase = 'rgb', 'flow' or 'gray' ")
        # pose
        pose_path = os.path.join(root, 'data/poses.h5', )
        pose_file = h5py.File(pose_path, 'r')
        self.pose = np.array(pose_file[rec[:-4] + '/poses'])
        h5_path = os.path.join(root, modality, rec+'.'+modality+'.h5')
        h5_file = h5py.File(h5_path, 'r', libver='latest', swmr=True)
        # print('Loaded: ' + rec+'.'+modality+'.h5')
        self.data = np.array(h5_file['data'])
        self.labels = np.array(h5_file['label'])
        self.frame_length = frame_length
        self.transform = transform
        self.phase = phase
        pose_file.close()
        h5_file.close()

    def __len__(self):
        return len(self.data) - self.frame_length

    def __getitem__(self, idx):
        new_idx = idx+self.frame_length
        # current frame time: new_idx-1
        # data.shape = (5, 224, 224, 3)
        data = self.data[idx:new_idx]
        pose = self.pose[idx:new_idx]
        # clip_data.shape = (224, 224, 15)
        clip_data = np.zeros((224, 224, self.num_channel * self.frame_length))
        clip_pose = np.zeros((16* self.frame_length), dtype=np.float32)
        
        for kk in range(self.frame_length):
            # take the first 8 joints and normalize to [0,1]
            # ipdb.set_trace()
            norm_pose = (pose[kk][:8] / np.array([1920, 1080])[None, :]).astype(np.float32)
            # resahpe to [x_0, y_0, x_1, y_1, ..., x_7, y_7]
            clip_pose[16*kk : 16*(kk+1)] = norm_pose.reshape(16)[:]
            # image data
            if self.phase == 'gray':
                clip_data[:, :, kk] = rgb2gray(data[kk][:])
            else:
                clip_data[:, :, self.num_channel*kk : self.num_channel*(kk+1)] = data[kk][:]
                
        # ipdb.set_trace()
        clip_pose = torch.from_numpy(clip_pose)
        clip_data = self.transform(clip_data)
        # next frame time: new_idx
        label = self.labels[new_idx]
        return clip_data, label, clip_pose


class Ikea_single(data.Dataset):
    '''Prepare for the inputs of single stream: rgb, flow, or gray.
    Read rgb/flow from the h5 files of each video. Used in train_fc.py
    '''
    def __init__(self, root, rec, phase, frame_length, clip_mean, clip_std):
        # modality: 'frames', 'flows'...
        if phase == 'rgb':
            modality = 'frames'
            self.num_channel = 3
        elif phase == 'flow':
            modality = 'flows'
            self.num_channel = 2
        elif phase == 'gray':
            modality = 'frames'
            self.num_channel = 1
        else:
            print("Only support phase = 'rgb', 'flow' or 'gray' ")
        
        # normalization
        self.clip_mean = np.array(clip_mean, dtype=np.float32)
        self.clip_std = np.array(clip_mean, dtype=np.float32)
        # Multiply by 255 to convert uint8 values later
        self.clip_mean *= 255
        self.clip_std *= 255

        # read data from h5 file
        h5_path = os.path.join(root, modality, rec+'.'+modality+'.h5')
        h5_file = h5py.File(h5_path, 'r', libver='latest', swmr=True)
        data = h5_file['data']
        self.data = np.empty(data.shape, data.dtype)
        data.read_direct(self.data)
        label = h5_file['label']
        self.labels = np.empty(label.shape, label.dtype)
        label.read_direct(self.labels)

        self.frame_length = frame_length
        # self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.data) - self.frame_length

    def __getitem__(self, idx):
        new_idx = idx+self.frame_length
        # current frame time: new_idx-1
        data = self.data[idx:new_idx]
        # clip_data = np.zeros((224, 224, self.num_channel * self.frame_length))
        nplanes = self.num_channel * self.frame_length
        clip_data = np.empty((nplanes, data.shape[1], data.shape[2]), np.float32)

        if self.phase == 'gray':
            # Convert RGB to gray: 0.299 R + 0.587 G + 0.114 B
            gray = np.array([0.299, 0.587, 0.114], dtype=np.float32)
            for kk in range(self.frame_length):
                # clip_data[:, :, kk] = rgb2gray(data[kk][:])
                clip_data[kk, :, :] = np.dot(data[kk], gray)
        else:
            for kk in range(self.frame_length):
                # clip_data[:, :, self.num_channel*kk : self.num_channel*(kk+1)] = data[kk][:]
                # Reshape (H x W x C) to (C x H x W)
                clip_data[self.num_channel*kk:self.num_channel*(kk+1), :, :] = (
                    data[kk][:].transpose(2, 0, 1))
        # ipdb.set_trace()
        # clip_data = self.transform(clip_data)
        # No need to divide clip_data by 255 because clip_mean and clip_std were
        # multiplied by 255.
        for i in range(nplanes):
            clip_data[i] -= self.clip_mean[i]
            clip_data[i] /= self.clip_std[i]
        clip_data = torch.from_numpy(clip_data)
        # next frame time: new_idx
        label = self.labels[new_idx]
        return clip_data, label


class Ikea_multi(data.Dataset):
    '''Prepare for the inputs of multi streams: rgb, flow and pose. 
    Read both rgb and flow from the h5 files of each video. Used in extract_feats.py 
    '''
    def __init__(self, root, rec, frame_length, rgb_trans, flow_trans):
        
        # rgb
        rgb_path = os.path.join(root, 'frames', rec+'.'+'frames'+'.h5')
        rgb_file = h5py.File(rgb_path, 'r', libver='latest', swmr=True)
        self.rgb = np.array(rgb_file['data'])
        self.rgb_trans = rgb_trans

        # flow
        flow_path = os.path.join(root, 'flows', rec+'.'+'flows'+'.h5')
        flow_file = h5py.File(flow_path, 'r', libver='latest', swmr=True)
        self.flow = np.array(flow_file['data'])
        self.flow_trans = flow_trans

        # pose
        pose_path = os.path.join(root, 'data/poses.h5')
        pose_file = h5py.File(pose_path, 'r')
        self.pose = np.array(pose_file[rec[:-4] + '/poses'])

        self.frame_length = frame_length
        self.labels = np.array(rgb_file['label'])

    def __len__(self):
        return len(self.flow) - self.frame_length

    def __getitem__(self, idx):
        new_idx = idx+self.frame_length
        rgb = self.rgb[idx:new_idx]
        flow = self.flow[idx:new_idx]
        pose = self.pose[idx:new_idx]
        
        clip_rgb = np.zeros((224, 224, 3 * self.frame_length))
        clip_flow = np.zeros((224, 224, 2 * self.frame_length))
        clip_pose = np.zeros((16* self.frame_length), dtype=np.float32)

        for kk in range(self.frame_length):
            norm_pose = (pose[kk][:8] / np.array([1920, 1080])[None, :]).astype(np.float32)
            # resahpe to [x_0, y_0, x_1, y_1, ..., x_7, y_7]
            clip_pose[16*kk : 16*(kk+1)] = norm_pose.reshape(16)[:]
            clip_rgb[:, :, 3*kk : 3*(kk+1)] = rgb[kk][:]
            clip_flow[:, :, 2*kk : 2*(kk+1)] = flow[kk][:]
            
        # ipdb.set_trace()
        clip_rgb = self.rgb_trans(clip_rgb)
        clip_flow = self.rgb_trans(clip_flow)
        clip_pose = torch.from_numpy(clip_pose)
        # next frame time: new_idx
        label = self.labels[new_idx]
        return clip_rgb, clip_flow, clip_pose, label


def rgb2gray(rgb):
    """convert rgb (224,224,3 ) to gray (224,224) image
    B' = 0.299 R + 0.587 G + 0.114 B
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]) # R G B




# ------------------------------------------------------------------------------
# First try:
# Read frames one by one from raw image flow data. Not from h5 files.
# Basic usage. There are only one dataload at every epoch.

class Ikea_frame(data.Dataset):  
    '''Single stream input. Rread rgb/flow one by one at every iteration.
    This class is very slow as it needs to load every rgb/flow from storage at 
    every epoch. Not used anymore.
    '''
    def __init__(self, root, rec, phase, frame_length, transform):

        # modality: 'frames', 'flows'...
        if phase == 'rgb':
            modality = 'frames'
        elif phase == 'flow':
            modality = 'flows'
        clips = make_dataset(root, rec, modality, frame_length)
        if len(clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                               "Check your data directory."))
        
        self.root = root
        self.rec = rec
        self.phase = phase
        self.frame_length = frame_length
        self.transform = transform
        self.clips = clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):

        file_dir, idx, label = self.clips[index]
        
        if self.phase == 'rgb':
            clip_input = np.zeros((360, 640, 3*self.frame_length))
            for kk in range(self.frame_length):
                file_path = os.path.join(file_dir, '%06d.jpg'%(idx-kk))
                clip = read_rgb(file_path)
                clip_input[:, :, 3*kk : 3*(kk+1)] = clip[:]
                # print(type(clip_input), clip_input.shape)
                # ipdb.set_trace()
        
        elif self.phase == 'flow':
            clip_input = np.zeros((360, 640, 2*self.frame_length))
            for kk in range(self.frame_length):
                file_path = os.path.join(file_dir, '%06d.flow'%(idx-kk))
                clip = read_flow(file_path)
                clip_input[:, :, 2*kk : 2*(kk+1)] = clip[:]

            
        clip_input = self.transform(clip_input)
        sample = (clip_input, label)
        return sample


def make_dataset(root_dir, rec_file, modality, frame_length):
    """Read training/testing record files and load their paths and labels to the
    memory. 
    """
    clips = []
    rec_path = os.path.join(root_dir, rec_file)
    # annotation list: dataset_public/ikea.../videos/2016-09-01/GOPRO115.MP4.30HZ.json
    
    with open(rec_path, 'r') as f_csv:
        annot_list = csv.reader(f_csv)

        for row in annot_list:
            # full path of json
            # TO DO: move out of this function
            sub_dirs = ''.join(row).split('/')
            annot_path = os.path.join(root_dir, sub_dirs[-3], sub_dirs[-2], sub_dirs[-1])
            img_dir = os.path.join(root_dir, modality, sub_dirs[-2], sub_dirs[-1][:-9] + modality) 
            # get abels from json file
            with open(annot_path, 'r') as f_json:
                annot = json.load(f_json)
                labels = annot['Y']
            # ipdb.set_trace()
            # use the previous frame_length frames to predict the folwing next frame
            # the number of flows is one less than that of rgb frames
            # label index starts from 0, but the file name starts from 000001
            for idx in range(frame_length, len(labels)):
                # img_path = os.path.join(img_dir, name_pattern % idx)
                clips.append((img_dir, idx, labels[idx]))
            
    return clips


def read_rgb(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        # ipdb.set_trace()
        print("Could not load file %s" % (img_path))
        sys.exit()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_flow(filename):
    FLO_TAG = 202021.25
    with open(filename, 'rb') as f:
        tag = np.fromfile(f, np.float32, count=1)
        
        if tag != FLO_TAG:
            sys.exit('Wrong tag. Invalid .flo file %s' %filename)
        else:
            w = int(np.fromfile(f, np.int32, count=1))
            h = int(np.fromfile(f, np.int32, count=1))
            #print 'Reading %d x %d flo file' % (w, h)
                
            data = np.fromfile(f, np.float32, count=2*w*h)

            # Reshape data into 3D array (columns, rows, bands)
            flow = np.resize(data, (h, w, 2))

    return flow


