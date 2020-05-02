import os, time, shutil, argparse, ipdb
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
from channel_resnet import channel_resnet18
from utils import AverageMeter, accuracy, save_checkpoint, adjust_learning_rate, parse_rec

parser = argparse.ArgumentParser(description="PyTorch Spatial ")
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 20)')
parser.add_argument('--phase', default='rgb', type=str, metavar='S', help='rgb, flow, pose')
parser.add_argument('--frame_length', default=5, type=int, metavar='N', help='the number of stacked frames (default: 5)')
parser.add_argument('--root_dir', default='../dataset_public/ikea-fa-release-data', type=str, metavar='S', help='dataset directory')
parser.add_argument('--model_folder', default='trained_models', type=str, metavar='S', help='model folder')
parser.add_argument('--train_rec', default='train_30fps.csv', type=str, metavar='S', help='training split')
parser.add_argument('--test_rec', default='test_30fps.csv', type=str, metavar='S', help='testing split')

def main():
    args = parser.parse_args()
    print(args)
    best_prec1 = 0
    class_num = 13

    # ipdb.set_trace()
    if args.phase == 'rgb':
        clip_mean = [0.485, 0.456, 0.406] * args.frame_length
        clip_std = [0.229, 0.224, 0.225] * args.frame_length
        in_channel = 3 * args.frame_length
        model_best = os.path.join(args.root_dir, args.model_folder, 'rgb_len5_model_best.pth.tar')

    elif args.phase == 'flow':
        clip_mean = [0.5, 0.5] * args.frame_length
        clip_std = [0.226, 0.226] * args.frame_length
        in_channel = 2 * args.frame_length
        model_best = os.path.join(args.root_dir, args.model_folder, 'flow_len5_model_best.pth.tar')

    test_recs = parse_rec(args.root_dir, args.test_rec)

    # -----------------------------------------------------------------------------
    # model
    feat_model = channel_resnet18(pretrained=True, in_channel=in_channel,
                                  num_classes=class_num)
    # loading trained models
    if os.path.isfile(model_best):
        print("=> loading checkpoint '{}'".format(model_best))
        checkpoint = torch.load(model_best)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        feat_model.load_state_dict(checkpoint['state_dict'])
        print(("=> best_prec1: '{}' (epoch {})"
              .format(best_prec1, start_epoch)))
    else:
        print("=> no checkpoint found.")
    # use cuda
    feat_model = feat_model.cuda()
    # evaluation mode
    feat_model.eval()
    criterion = nn.CrossEntropyLoss().cuda()

    # ----------------------------------------------------------------------------
    #Training & testing
    # ipdb.set_trace()
    test_acc = AverageMeter('test_acc', ':6.2f')
    test_loss = AverageMeter('test_loss', ':.4e')
    for idx in range(len(test_recs)):
        test_set = data.Ikea(root = args.root_dir, rec = test_recs[idx],
                    phase = args.phase, frame_length = args.frame_length,
                    clip_mean=clip_mean, clip_std=clip_std)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=True)
        with torch.no_grad():
            acc, loss = test(test_loader, feat_model, criterion, args)
        test_acc.update(acc)
        test_loss.update(loss)
        print(' Testing on {rec}\t Acc: {acc:.3f}\t Loss: {loss:.4f} '
            .format(rec=test_recs[idx], acc=acc, loss=loss))
    
    print(' *Acc: {acc:.3f}\t Loss: {loss:.4f}'
            .format(acc=test_acc.avg, loss=test_loss.avg))

def test(test_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    # progress = tqdm(test_loader)
    for i, (data,label) in enumerate(test_loader):
        
        label = label.cuda(non_blocking=True)
        data = data.cuda(non_blocking=True)

        # compute output
        output = model(data)
        loss = criterion(output, label)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, label, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1[0], data.size(0))
    return top1.avg, losses.avg

if __name__ == '__main__':
	main()
