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
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=80, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_steps', default=30, type=int, metavar='LRSteps', help='epochs to decay lr by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')

parser.add_argument('--root_dir', default='../data/OAD/', type=str, metavar='S', help='dataset directory')
parser.add_argument('--model_folder', default='trained_models', type=str, metavar='S', help='model folder')
parser.add_argument('--model_exp', default='fc', type=str, metavar='S', help='model experiments')
parser.add_argument('--log_folder', default='logs', type=str, metavar='S', help='logs for tensorboard')

parser.add_argument('--mode', default='rgb', type=str, metavar='S', help='rgb, flow, pose')
parser.add_argument('--frame_length', default=5, type=int, metavar='N', help='the number of stacked frames (default: 5)')
parser.add_argument('--fix', default=False, type=bool, metavar='B', help='whether to fix weights')
parser.add_argument('--pose_flag', default='n', type=str, metavar='S', help='whether to add pose feature')


def train(train_loader, model, criterion, optimizer, epoch, args):
    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    #switch to train mode
    model.train()
    # progress = tqdm(train_loader)
    # for i, (data,label) in enumerate(train_loader):

    # use DataPrefetcher() to speed up dataloader
    i = 0
    prefetcher = DataPrefetcher(train_loader)
    data, label = prefetcher.next()
    while data is not None:
        i += 1
        # ipdb.set_trace()
        
        data = data.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # compute output
        output = model(data)
        loss = criterion(output, label)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, label, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # read next batch
        data, label = prefetcher.next()
    
    return top1.avg, losses.avg


def validate(test_loader, model, criterion, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    # progress = tqdm(test_loader)
    # for i, (data,label) in enumerate(test_loader):

    # use DataPrefetcher() to speed up dataloader
    i = 0
    prefetcher = DataPrefetcher(test_loader)
    data, label = prefetcher.next()
    while data is not None:
        i += 1
        
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

        # read next batch
        data, label = prefetcher.next()

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
    log_dir = os.path.join(args.root_dir, args.log_folder, exp_name, log_name)

    model_dir = os.path.join(args.root_dir, args.model_folder, exp_name, log_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    ckpt_path = os.path.join(model_dir, 'checkpoint.pth.tar')
    best_path = os.path.join(model_dir, 'model_best.pth.tar')
    
    # check if the experiment has been performed
    if os.path.isdir(log_dir):
        print("Exp '"+ exp_name +'/'+ log_name + "'", 'has already been performed.')
        print('Please manually delete the following dirs if you want to re-run it:')
        print(log_dir)
        print(model_dir)
        sys.exit(0)
    
    # --------------------------------------------------------------------------
    # mode config
    if args.mode == 'rgb':
        data_name = 'rgbs.h5'
        in_channel = 3 * args.frame_length
    elif args.mode == 'flow':
        data_name = 'flows.h5'
        in_channel = 2 * args.frame_length
    elif args.mode == 'pose':
        data_name = 'poses.h5'
        in_channel = 2 * args.frame_length
    else:
        print("Only support mode = 'rgb', 'flow' or 'pose' ")

    # load all data
    file_path = os.path.join(args.root_dir, 'group_data', data_name)
    dataset, labels = data.load_dataset(file_path, args.mode, train_flag=True)

    # --------------------------------------------------------------------------
    # Build k channel network with pre-trained weight
    model = resnet.resnet18(pretrained=True, in_channel=in_channel, 
                            num_classes=class_num, fix=args.fix)
    # convert model to multi gpus
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model).cuda()
    
    #Loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # Optimizer and lr_scheduler
    if args.fix:
        optim_para = filter(lambda p:p.requires_grad, model.parameters())
    else:
        optim_para = model.parameters()
    optimizer = torch.optim.SGD(optim_para, momentum=args.momentum,
                                lr=args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True
    
    # --------------------------------------------------------------------------
    #Training & testing
    # use tensorboard to visualize
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    
    for epoch in range(1, args.epochs+1):
        adjust_learning_rate(optimizer, epoch, args)
        # count in the results of all data loaders
        train_acc = AverageMeter('train_acc', ':6.2f')
        test_acc = AverageMeter('test_acc', ':6.2f')
        train_loss = AverageMeter('train_loss', ':.4e')
        test_loss = AverageMeter('test_loss', ':.4e')
        
        # traing one epoch on all test_loaders
        print('Epoch:[{0}/{1}][training stage]'.format(epoch, args.epochs))
        # shuffle train_loaders
        shffle_num = np.random.permutation( np.arange(0, 30) )
        for idx in tqdm(shffle_num):

            train_set = data.OAD_single(root=args.root_dir, mode=args.mode, 
                        frame_length=args.frame_length, seq_data = dataset[idx], 
                        seq_label=labels[idx])
            train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers, pin_memory=True)
            acc, loss = train(train_loader, model, criterion, optimizer, epoch, args)
            train_acc.update(acc)
            train_loss.update(loss)
            print(' Training on seq #{idx}\t Acc: {acc:.3f}\t Loss: {loss:.4f} '
                .format(idx=idx, acc=acc, loss=loss))

        # testing one epoch on all test_loaders
        print('Epoch:[{0}/{1}][validation stage]'.format(epoch, args.epochs))
        for idx in np.arange(30, 50):
            test_set = data.OAD_single(root=args.root_dir, mode=args.mode, 
                        frame_length=args.frame_length, seq_data = dataset[idx], 
                        seq_label=labels[idx])
            test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)
            acc, loss = validate(test_loader, model, criterion, epoch, args)
            test_acc.update(acc)
            test_loss.update(loss)
            print(' Testing on seq #{idx}\t Acc: {acc:.3f}\t Loss: {loss:.4f} '
                .format(idx=idx, acc=acc, loss=loss))
       

        # write to tensorboard
        writer.add_scalars('losses', {'train_loss' : train_loss.avg,
                                      'test_loss' : test_loss.avg}, epoch)
        writer.add_scalars('Acc@1', {'train_acc' : train_acc.avg,
                                      'test_acc' : test_acc.avg}, epoch)
        writer.close()

        # remember best prec@1 and save checkpoint
        is_best = test_acc.avg > best_prec1
        best_prec1 = max(test_acc.avg, best_prec1)
        print(' *Acc: {acc:.3f}\t Loss: {loss:.4f}\t Acc_best:{best:.3f}'
            .format(acc=test_acc.avg, loss=test_loss.avg, best=best_prec1.item()))
        save_checkpoint({
            'epoch': epoch,
            'arch': 'resnet18',
            'state_dict': model.state_dict(),
            # 'state_dict': model.module.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, ckpt_path, best_path)

