import os, time, shutil, argparse, ipdb, sys, h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import data, video_transforms
from data import DataPrefetcher
from nets import resnet
from utils import str2bool, accuracy, save_checkpoint, adjust_learning_rate
from utils import AverageMeter, FeatureExtractor

parser = argparse.ArgumentParser(description="PyTorch Spatial ")
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 128)')

parser.add_argument('--root_dir', default='../data/OAD/', type=str, metavar='S', help='dataset directory')
parser.add_argument('--model_folder', default='trained_models', type=str, metavar='S', help='model folder')
parser.add_argument('--model_exp', default='fc', type=str, metavar='S', help='model experiments')
parser.add_argument('--log_folder', default='logs', type=str, metavar='S', help='logs for tensorboard')

parser.add_argument('--frame_length', default=5, type=int, metavar='N', help='the number of stacked frames (default: 5)')
parser.add_argument('--fix', default=False, type=bool, metavar='B', help='whether to fix weights')
parser.add_argument('--pose_flag', default='n', type=str, metavar='S', help='whether to add pose feature')


def build_path(mode):
    exp_name = '{}_len{}'.format(args.model_exp, args.frame_length)
    log_name = "{}_{}_pose{}".format(exp_name, mode, str2bool(args.pose_flag))
    
    model_dir = os.path.join(args.root_dir, args.model_folder, exp_name, log_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    ckpt_path = os.path.join(model_dir, 'checkpoint.pth.tar')
    best_path = os.path.join(model_dir, 'model_best.pth.tar')
    return ckpt_path, best_path


def build_model(num_class, in_channel, ckpt_path):
    '''Build trained models.
    ''' 
    model = resnet.resnet18(pretrained=False, in_channel=in_channel, 
                            num_classes=num_class, fix=args.fix)
    model = torch.nn.DataParallel(model)
    # loading trained models
    if os.path.isfile(ckpt_path):
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print(("=> best_prec1: '{}' (epoch {})"
              .format(best_prec1, epoch)))
    else:
        print("=> no checkpoint found.")
    return model.cuda().eval()


def build_loader(args, idx, mode):
    data_set = data.OAD_single_test(root=args.root_dir, mode=mode, 
                    frame_length=args.frame_length, seq_idx=idx)
    data_loader = DataLoader(dataset=data_set, batch_size=args.batch_size,
                    shuffle=False, num_workers=args.workers, pin_memory=False)
    return data_loader


def extract_pose(pose_loader):
    nframes = len(pose_loader)
    pose_feats = torch.zeros([nframes, 150], dtype=torch.float)
    for i, (pose, label) in enumerate(pose_loader):
        pose_feats[i, :] = pose[0][:]
    return pose_feats


def extract_feat(data_loader, extractor, feat_size): 
    nframes = len(data_loader)
    feats = torch.zeros([nframes, feat_size], dtype=torch.float)
    labels = torch.zeros(nframes, dtype=torch.int8)
    
    # use DataPrefetcher() to speed up dataloader
    i = 0
    prefetcher = DataPrefetcher(data_loader)
    data, label = prefetcher.next()

    while data is not None:
        # ipdb.set_trace()
        data = data.cuda(non_blocking=True)
        feat = extractor(data)[0]
        feat = feat.view(feat.size(0), -1)  # feat.shape is (1, 512)
        feats[i, :] = feat.cpu().squeeze()[:]
        labels[i] = label

        # read next batch
        i += 1
        data, label = prefetcher.next()

    return feats, labels


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    num_class = 11
    feat_size = 512
    save_path = os.path.join(args.root_dir, 'group_data', 'feats.h5')
    if os.path.exists(save_path):
        os.remove(save_path)
        print('Delete', save_path)
    # ipdb.set_trace()
    
    # --------------------------------------------------------------------------
    # model config
    rgb_model_path, _ = build_path('rgb')
    rgb_channel = 3 * args.frame_length
    rgb_model = build_model(num_class, rgb_channel, rgb_model_path)

    flow_model_path, _ = build_path('flow')
    flow_channel = 2 * args.frame_length
    flow_model = build_model(num_class, flow_channel, flow_model_path)
    
    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True
    
    # --------------------------------------------------------------------------
    # extract feats
    exact_list=["avgpool"]  # features before dropout
    for idx in range(0, 59):

        rgb_loader = build_loader(args, idx, 'rgb')
        flow_loader = build_loader(args, idx, 'flow')
        pose_loader = build_loader(args, idx, 'pose')
        pose_1080p_loader = build_loader(args, idx, 'pose_1080p')
        
        print('Start to extract features...')
        with torch.no_grad():
            # extract deep features
            rgb_extractor=FeatureExtractor(rgb_model, exact_list)
            rgb_feats, labels = extract_feat(rgb_loader, rgb_extractor, feat_size)
            
            flow_extractor=FeatureExtractor(flow_model, exact_list)
            flow_feats, _ = extract_feat(flow_loader, flow_extractor, feat_size)
            
            pose_feats = extract_pose(pose_loader)
            pose_feats_1080p = extract_pose(pose_1080p_loader)
            # combine all feats into one dict per sequence, but h5py can only 
            # save numpy style data
            # feats = {'rgb': rgb_feats, 'flow': flow_feats, 'label': labels,
            #          'pose': pose_feats, 'pose_1080p': pose_feats_1080p}
            
            # save data to h5 file
            # ipdb.set_trace()
            compression = 32001
            compression_opts = (0, 0, 0, 0, 9, False, 1)
            f = h5py.File(save_path, 'a')
            grp = f.create_group(str(idx).zfill(2))
            grp.create_dataset('rgb', data=rgb_feats)
            grp.create_dataset('flow', data=flow_feats)
            grp.create_dataset('pose', data=pose_feats)
            grp.create_dataset('pose_1080p', data=pose_feats_1080p)
            grp.create_dataset('label', data=labels)
            f.close()
        