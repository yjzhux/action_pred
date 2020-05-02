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
from nets import resnet, lstm
from utils import accuracy, seq_accuracy, save_checkpoint, adjust_learning_rate, parse_rec, str2bool
from utils import AverageMeter, FeatureExtractor, SeqCrossEntropyLoss, SeqSmoothL1Loss

parser = argparse.ArgumentParser(description="PyTorch Spatial ")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--epochs', default=80, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_steps', default=30, type=int, metavar='LRSteps', help='epochs to decay lr by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--print_freq', default=20, type=int, metavar='N', help='print frequency (default: 20)')
parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--phase', default='rgb', type=str, metavar='S', help='rgb, flow, pose')
parser.add_argument('--future_length', default=0, type=int, metavar='N', help='the number of future frames')
parser.add_argument('--root_dir', default='../dataset_public/ikea-fa-release-data', type=str, metavar='S', help='dataset directory')
parser.add_argument('--model_folder', default='trained_models', type=str, metavar='S', help='model folder')
parser.add_argument('--log_folder', default='logs', type=str, metavar='S', help='logs for tensorboard')
parser.add_argument('--feat_file', default='data/feats_len5.h5', type=str, metavar='S', help='saved feature file')
parser.add_argument('--train_rec', default='train_30fps.csv', type=str, metavar='S', help='training split')
parser.add_argument('--test_rec', default='test_30fps.csv', type=str, metavar='S', help='testing split')
parser.add_argument('--fix', default='n', type=str, metavar='S', help='whether to fix weights')
parser.add_argument('--pose_flag', default='n', type=str, metavar='S', help='whether to add pose feature')

def main():
    args = parser.parse_args()
    print(args)
    best_prec1 = 0
    class_num = 13
    log_name = "{}_pose{}_future{}_lstm".format(args.phase, str2bool(args.pose_flag), args.future_length)
    ckpt_path = os.path.join(args.root_dir, args.model_folder, log_name + '_checkpoint.pth.tar')
    best_path = os.path.join(args.root_dir, args.model_folder, log_name + '_model_best.pth.tar')
    log_dir = os.path.join(args.root_dir, args.log_folder, log_name)
    hidden_size = 128

    pose_flag = str2bool(args.pose_flag)
    future = int(args.future_length / 5)
    if args.phase == 'rgb' or args.phase == 'flow':
        input_size = 512 
    elif args.phase == 'rgb_flow':
        input_size = 512 * 2 
    elif args.phase == 'pose':
        input_size = 80  
        pose_flag = False
    else:
        print("Only support phase = 'rgb', 'flow' or 'pose' ")
    # ipdb.set_trace()
    if pose_flag:
        # resnet feature + coordinates of the first 8 joints
        input_size += 80 
    
    train_recs = parse_rec(args.root_dir, args.train_rec)
    train_set = data.Ikea_feat_cls_reg(root = args.root_dir, rec = train_recs, future_length=args.future_length,
                    phase = args.phase, pose_flag = pose_flag, feat_file = args.feat_file)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, pin_memory=True)
    
    test_recs = parse_rec(args.root_dir, args.test_rec)
    test_set = data.Ikea_feat_cls_reg(root = args.root_dir, rec = test_recs, future_length=args.future_length,
                    phase = args.phase, pose_flag = pose_flag, feat_file = args.feat_file)
    test_loader = DataLoader(dataset=test_set, batch_size=1,
                    shuffle=False, num_workers=args.workers, pin_memory=True)

    # ----------------------------------------------------------------------------
    # lstm model to be trained
    model = lstm.LSTMclsReg(input_size, hidden_size, class_num, n_layers=1)
    model = model.cuda()
    #Loss function
    criterion = {0: SeqCrossEntropyLoss().cuda(), 1: SeqSmoothL1Loss().cuda()}
    # Optimizer and lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True
    
    # ----------------------------------------------------------------------------
    #Training & testing
    # use tensorboard to visualize
    # ipdb.set_trace()
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    # add model as graph tensorboard
    # writer.add_graph(model)
    
    for epoch in range(1, args.epochs+1):
        
        # traing one epoch on all test_loaders
        adjust_learning_rate(optimizer, epoch, args)
        print('Epoch:[{0}/{1}][training stage]'.format(epoch, args.epochs))
        train_acc, train_loss, train_loss_reg = train(train_loader, model, criterion, optimizer, epoch, args)
        
        # testing one epoch on all test_loaders
        print('Epoch:[{0}/{1}][validation stage]'.format(epoch, args.epochs))
        test_acc, test_loss, test_loss_reg = validate(test_loader, model, criterion, epoch, args)
       
        # write to tensorboard
        writer.add_scalars('RegLoss', {'train_loss_reg' : train_loss_reg,
                                      'test_loss_reg' : test_loss_reg}, epoch)
        writer.add_scalars('Loss', {'train_loss' : train_loss,
                                      'test_loss' : test_loss}, epoch)
        writer.add_scalars('Acc', {'train_acc' : train_acc,
                                      'test_acc' : test_acc}, epoch)
        writer.close()

        # remember best prec@1 and save checkpoint
        is_best = test_acc > best_prec1
        best_prec1 = max(test_acc, best_prec1)
        print(' *Acc: {acc:.3f}\t Loss: {loss:.4f}\t Acc_best:{best:.3f}'
            .format(acc=test_acc, loss=test_loss, best=best_prec1))
        save_checkpoint({
            'epoch'     : epoch,
            'arch'      : 'lstm',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, ckpt_path, best_path)
    


def train(train_loader, model, criterion, optimizer, epoch, args):
    
    losses_reg = AverageMeter('Loss', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    #switch to train mode
    model.train()
    for i, (data, label, pose) in enumerate(tqdm(train_loader)):
        label = label.long().cuda(non_blocking=True)
        data = data.cuda(non_blocking=True)
        pose = pose.cuda(non_blocking=True)
        
        # compute output
        # ipdb.set_trace()
        class_out, pose_out = model(data)
        loss_cls = criterion[0](class_out, label)
        loss_reg = criterion[1](pose_out, pose)
        loss = loss_cls + loss_reg

        # measure accuracy and record loss
        acc = seq_accuracy(class_out, label, topk=(1, ))
        losses.update(loss.item(), data.size(1))
        losses_reg.update(loss_reg.item(), data.size(1))
        top1.update(acc[0].item(), data.size(1))
        if (i+1) % args.print_freq == 0:
            print(' Epochs [{0}/{1}]\t Acc: {acc:.3f}\t Loss: {loss:.4f} '
                .format(epoch, args.epochs, acc=top1.avg, loss=losses_reg.avg))

        # compute gradient and do SGD step
        # Clears existing gradients from previous epoch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return top1.avg, losses.avg, losses_reg.avg


def validate(test_loader, model, criterion, epoch, args):
    losses_reg = AverageMeter('Loss', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (data, label, pose) in enumerate(test_loader):
            
            label = label.long().cuda(non_blocking=True)
            data = data.cuda(non_blocking=True)
            pose = pose.cuda(non_blocking=True)
        
            # compute output
            # ipdb.set_trace()
            class_out, pose_out = model(data)
            loss_cls = criterion[0](class_out, label)
            loss_reg = criterion[1](pose_out, pose)
            loss = loss_cls + loss_reg

            # measure accuracy and record loss
            acc = seq_accuracy(class_out, label, topk=(1, ))
            losses.update(loss.item(), data.size(1))
            losses_reg.update(loss_reg.item(), data.size(1))
            top1.update(acc[0].item(), data.size(1))

            # print accuracy of each sequence
            print(' Sequence {0}:\t Acc: {acc:.3f}\t Loss: {loss:.4f} '
                .format(i+1, acc=top1.avg, loss=losses_reg.avg))

    return top1.avg, losses.avg, losses_reg.avg


if __name__ == '__main__':
	main()
