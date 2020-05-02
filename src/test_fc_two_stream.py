import os, time, shutil, argparse, ipdb, h5py
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
from nets import resnet
from utils import accuracy, save_checkpoint, adjust_learning_rate, parse_rec
from utils import AverageMeter, FeatureExtractor

parser = argparse.ArgumentParser(description="PyTorch Spatial ")
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 20)')
parser.add_argument('--frame_length', default=5, type=int, metavar='N', help='the number of stacked frames (default: 5)')
parser.add_argument('--root_dir', default='../dataset_public/ikea-fa-release-data', type=str, metavar='S', help='dataset directory')
parser.add_argument('--model_folder', default='trained_models', type=str, metavar='S', help='model folder')
parser.add_argument('--train_rec', default='train_30fps.csv', type=str, metavar='S', help='training split')
parser.add_argument('--test_rec', default='test_30fps.csv', type=str, metavar='S', help='testing split')

def main():
    args = parser.parse_args()
    print(args)
    num_class = 13
    feat_size = 512
    # open the h5 file to save all feature data to one file
    save_path = os.path.join(args.root_dir, 'data/feats.h5')
    # if os.path.exists(save_path):
    #     os.remove(save_path)
    #     print('Deleted data/feats.h5')

    # transform setting
    rgb_mean = [0.485, 0.456, 0.406] * args.frame_length
    rgb_std = [0.229, 0.224, 0.225] * args.frame_length
    rgb_channel = 3 * args.frame_length
    rgb_model_path = os.path.join(args.root_dir, args.model_folder, 'rgb_len5_model_best.pth.tar')
    rgb_trans=build_transform(rgb_mean,rgb_std)

    flow_mean = [0.5, 0.5] * args.frame_length
    flow_std = [0.226, 0.226] * args.frame_length
    flow_channel = 2 * args.frame_length
    flow_model_path = os.path.join(args.root_dir, args.model_folder, 'flow_len5_model_best.pth.tar')
    flow_trans=build_transform(flow_mean,flow_std)

    train_recs = parse_rec(args.root_dir, args.train_rec)
    test_recs = parse_rec(args.root_dir, args.test_rec)
    

    # -----------------------------------------------------------------------------
    # model
    rgb_model = build_model(num_class, rgb_channel, rgb_model_path)
    flow_model = build_model(num_class, flow_channel, flow_model_path)
    criterion = nn.CrossEntropyLoss().cuda()

    # ----------------------------------------------------------------------------
    # testing
    # ipdb.set_trace()
    test_acc = AverageMeter('test_acc', ':6.2f')
    test_loss = AverageMeter('test_loss', ':.4e')
    for idx in range(len(test_recs)):

        vid_middle = test_recs[idx][:-4]
        test_set = data.Ikea_multi(root = args.root_dir, rec = test_recs[idx],
                    frame_length = args.frame_length,
                    rgb_trans = rgb_trans, flow_trans = flow_trans)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=True)
        with torch.no_grad():
            # testing two stream
            acc, loss = test(test_loader, rgb_model, flow_model, criterion, args)
            test_acc.update(acc)
            test_loss.update(loss)
            print(' Testing on {rec}\t Acc: {acc:.3f}\t Loss: {loss:.4f} '
                .format(rec=test_recs[idx], acc=acc, loss=loss))

    
    print(' *Acc: {acc:.3f}\t Loss: {loss:.4f}'
            .format(acc=test_acc.avg, loss=test_loss.avg))


def build_model(num_class, in_channel, model_best):
    '''Build trained models.
    ''' 
    model = resnet.resnet18(pretrained=False, in_channel=in_channel, 
                            num_classes=num_class)
    # loading trained models
    if os.path.isfile(model_best):
        print("=> loading checkpoint '{}'".format(model_best))
        checkpoint = torch.load(model_best)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print(("=> best_prec1: '{}' (epoch {})"
              .format(best_prec1, start_epoch)))
    else:
        print("=> no checkpoint found.")
    # use cuda
    model = model.cuda()
    
    return model.eval()    # evaluation mode


def build_transform(clip_mean, clip_std):
    transform=video_transforms.Compose([
                video_transforms.ToTensor(),
                video_transforms.Normalize(mean=clip_mean,std=clip_std)])    
    return transform


def test(test_loader, rgb_model, flow_model, criterion, args):
    '''Test two streams modles.
    '''
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # progress = tqdm(test_loader)
    for i, (rgb, flow, pose, label) in enumerate(test_loader):
        
        label = label.cuda(non_blocking=True)
        rgb = rgb.cuda(non_blocking=True)
        flow = flow.cuda(non_blocking=True)

        # compute output
        rgb_output = rgb_model(rgb)
        flow_output = flow_model(flow)
        output = (rgb_output + flow_output)/2
        loss = criterion(output, label)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, label, topk=(1, 5))
        losses.update(loss.item(), rgb.size(0))
        top1.update(prec1[0], rgb.size(0))
    return top1.avg, losses.avg

if __name__ == '__main__':
	main()
