import os, time, shutil, argparse, ipdb, sys
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
from utils import AverageMeter, accuracy, save_checkpoint, adjust_learning_rate
from utils import str2bool

parser = argparse.ArgumentParser(description="PyTorch Spatial ")
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 128)')

parser.add_argument('--root_dir', default='../dataset_public/OAD/', type=str, metavar='S', help='dataset directory')
parser.add_argument('--model_folder', default='trained_models', type=str, metavar='S', help='model folder')
parser.add_argument('--model_exp', default='fc', type=str, metavar='S', help='model experiments')
parser.add_argument('--log_folder', default='logs', type=str, metavar='S', help='logs for tensorboard')

parser.add_argument('--mode', default='rgb', type=str, metavar='S', help='rgb, flow, pose')
parser.add_argument('--frame_length', default=2, type=int, metavar='N', help='the number of stacked frames (default: 5)')
parser.add_argument('--fix', default=False, type=bool, metavar='B', help='whether to fix weights')
parser.add_argument('--pose_flag', default='n', type=str, metavar='S', help='whether to add pose feature')


def test(test_loader, model, criterion, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    for i, (data,label) in enumerate(test_loader):

    # # use DataPrefetcher() to speed up dataloader
    # i = 0
    # prefetcher = DataPrefetcher(test_loader)
    # data, label = prefetcher.next()
    # while data is not None:
    #     i += 1
        
        label = label.cuda(non_blocking=True)
        data = data.cuda(non_blocking=True)

        # compute output
        output = model(data)
        loss = criterion(output, label)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, label, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        # # read next batch
        # data, label = prefetcher.next()

    return top1.avg, losses.avg


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    best_prec1 = 0
    class_num = 11
    # ipdb.set_trace()
    
    # --------------------------------------------------------------------------
    # path config
    exp_name = '{}_len{}'.format(args.model_exp, args.frame_length)
    log_name = "{}_{}_pose{}".format(exp_name, args.mode, str2bool(args.pose_flag))
    
    model_dir = os.path.join(args.root_dir, args.model_folder, exp_name, log_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    ckpt_path = os.path.join(model_dir, 'checkpoint.pth.tar')
    best_path = os.path.join(model_dir, 'model_best.pth.tar')
    
    # --------------------------------------------------------------------------
    # mode config
    if args.mode == 'rgb':
        in_channel = 3 * args.frame_length
    elif args.mode == 'flow':
        in_channel = 2 * args.frame_length
    elif args.mode == 'pose':
        in_channel = 2 * args.frame_length
    else:
        print("Only support mode = 'rgb', 'flow' or 'pose' ")

    # --------------------------------------------------------------------------
    # Build k channel network with pre-trained weight
    model = resnet.resnet18(pretrained=False, in_channel=in_channel, 
                            num_classes=class_num, fix=args.fix)
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
    # use cuda
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True
    
    # --------------------------------------------------------------------------
    #Training & testing
    test_acc = AverageMeter('test_acc', ':6.2f')
    test_loss = AverageMeter('test_loss', ':.4e')
    
    for idx in range(30, 50):
        test_set = data.OAD_single_test(root=args.root_dir, mode=args.mode, 
                    frame_length=args.frame_length, seq_idx=idx)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=True)
        acc, loss = test(test_loader, model, criterion, epoch, args)
        test_acc.update(acc)
        test_loss.update(loss)
        print(' Testing on seq #{idx}\t Acc: {acc:.3f}\t Loss: {loss:.4f} '
            .format(idx=idx, acc=acc, loss=loss))

    print(' *Acc: {acc:.3f}\t Loss: {loss:.4f}'
            .format(acc=test_acc.avg, loss=test_loss.avg))   