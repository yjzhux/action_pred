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

import data
from nets import lstm
from utils import accuracy, save_checkpoint, adjust_learning_rate, str2bool
from utils import AverageMeter, SeqCrossEntropyLoss, SeqSmoothL1Loss


parser = argparse.ArgumentParser(description="PyTorch Spatial ")
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=80, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_steps', default=30, type=int, metavar='LRSteps', help='epochs to decay lr by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')

parser.add_argument('--root_dir', default='../data_public/OAD/', type=str, metavar='S', help='dataset directory')
parser.add_argument('--model_folder', default='trained_models', type=str, metavar='S', help='model folder')
parser.add_argument('--model_exp', default='seq2seq', type=str, metavar='S', help='model experiments')
parser.add_argument('--log_folder', default='logs', type=str, metavar='S', help='logs for tensorboard')

parser.add_argument('--print_freq', default=20, type=int, metavar='N', help='print frequency (default: 20)')
parser.add_argument('--log_freq', default=5, type=int, metavar='N', help='tensorboard log frequency (default: 5)')
parser.add_argument('--pw', default=100, type=float, metavar='M', help='the weight of pose regression loss.')
parser.add_argument('--mode', default='rgb', type=str, metavar='S', help='rgb, flow, pose')
parser.add_argument('--future_length', default=5, type=int, metavar='N', help='the number of future frames')
parser.add_argument('--fix', default=False, type=bool, metavar='B', help='whether to fix weights')
parser.add_argument('--pose_1080p_flag', default='y', type=str, metavar='S', help='whether to use pose extracted from 1080p images by openpose')
parser.add_argument('--pose_flag', default='n', type=str, metavar='S', help='whether to add pose feature')
parser.add_argument('--task', default='joint', type=str, metavar='S', help="task: 'class', 'pose', 'joint'")


def train(train_loader, model, criterion, optimizer, epoch, args):
    
    losses = AverageMeter('Loss', ':.4e')
    losses_reg = AverageMeter('Loss', ':.4e')
    top1 = [AverageMeter('Acc@{}'.format(k), ':6.2f') for k in range(args.future_length)]
    len_acc = AverageMeter('Len_acc', ':6.2f')
    
    #switch to train mode
    model.train()
    # progress = tqdm(train_loader)
    for i, (data,label,pose) in enumerate(train_loader):
        # ipdb.set_trace()
        data = data.cuda(non_blocking=True)
        label = label.long().cuda(non_blocking=True)
        pose = pose.cuda(non_blocking=True)

        # compute output
        class_out, pose_out = model(data)

        loss = 0
        loss_step = 0
        length_acc = 0
        for k in range(args.future_length):
            # compute loss
            # ipdb.set_trace()
            loss_cls = criterion[0](class_out[k], label[0, k, :])
            loss_reg = criterion[1](pose_out[k], pose[0, k, :])
            # loss += (loss_cls + args.pw*loss_reg)
            # loss_step += loss_reg

            if args.task == 'class':
                loss += loss_cls
            elif args.task == 'pose':
                loss += loss_reg
            elif args.task == 'joint':
                loss += (loss_cls + args.pw*loss_reg)
            else:
                raise Exception("Unsupport task!", args.task)
            loss_step += loss_reg

            # measure accuracy 
            acc = accuracy(class_out[k], label[0, k, :], topk=(1, ))
            top1[k].update(acc[0].item(), label.size(2))
            length_acc += acc[0].item()
        
        losses.update(loss.item()/args.future_length, label.size(2))
        losses_reg.update(loss_step.item()/args.future_length, label.size(2))
        len_acc.update(length_acc/args.future_length, label.size(2))
        if (i+1) % args.print_freq == 0:
            # ipdb.set_trace()
            print(' Epochs [{0}/{1}]\t Len_acc: {acc:.3f}\t Loss_reg: {loss:.4f} '
                .format(epoch, args.epochs, acc=len_acc.avg, loss=losses_reg.avg))

        # compute gradient and do SGD step
        # Clears existing gradients from previous epoch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return top1, len_acc.avg, losses.avg, losses_reg.avg


def validate(test_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4e')
    losses_reg = AverageMeter('Loss', ':.4e')
    top1 = [AverageMeter('Acc@{}'.format(k), ':6.2f') for k in range(args.future_length)]
    len_acc = AverageMeter('Len_acc', ':6.2f')

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

            loss = 0
            loss_step = 0
            length_acc = 0
            for k in range(args.future_length):
                # compute loss
                # loss += criterion[0](output[k], label[0, k, :])
                loss_cls = criterion[0](class_out[k], label[0, k, :])
                loss_reg = criterion[1](pose_out[k], pose[0, k, :])
                # loss += (loss_cls + args.pw * loss_reg)
                # loss_step += loss_reg

                if args.task == 'class':
                    loss += loss_cls
                elif args.task == 'pose':
                    loss += loss_reg
                elif args.task == 'joint':
                    loss += (loss_cls + args.pw*loss_reg)
                else:
                    raise Exception("Unsupport task!", args.task)
                loss_step += loss_reg

                # measure accuracy 
                acc = accuracy(class_out[k], label[0, k, :], topk=(1, ))
                top1[k].update(acc[0].item(), label.size(2))
                length_acc += acc[0].item()
            
            losses.update(loss.item()/args.future_length, label.size(2))
            losses_reg.update(loss_step.item()/args.future_length, label.size(2))
            # len_acc.update(length_acc/args.future_length, label.size(2))
            # average all predited timesteps
            avg_acc = length_acc/args.future_length
            len_acc.update(avg_acc, label.size(2))

            # print accuracy of each sequence
            print(' Sequence {0}:\t Len_acc: {acc:.3f}\t Loss_reg: {loss:.4f} '
                .format(i+1, acc=avg_acc, loss=losses_reg.avg))

    return top1, len_acc.avg, losses.avg, losses_reg.avg


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    best_prec1 = 0
    class_num = 11
    hid_dim = 128
    feat_size = 512
    num_joint = 15
    pose_size = 2 * num_joint * args.future_length
    data_name = 'feats.h5'
    # ipdb.set_trace()
    
    # --------------------------------------------------------------------------
    # path config
    exp_name = '{}_future{}_{}'.format(args.model_exp, args.future_length, args.task)
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
    # encoder input config according to args.phase
    pose_1080p_flag = str2bool(args.pose_1080p_flag)
    pose_flag = str2bool(args.pose_flag)
    if args.mode == 'rgb' or args.mode == 'flow':
        input_size = feat_size 
    elif args.mode == 'rgb_flow':
        input_size = 2 * feat_size
    elif args.mode == 'pose':
        input_size = pose_size
        pose_flag = False
    else:
        print("Only support mode = 'rgb', 'flow', 'pose' or 'rgb_flow'")
    # ipdb.set_trace()
    if pose_flag:
        # resnet feature + coordinates of the joints
        input_size += pose_size  

    # decoder input config according to args.task
    if args.task == 'class':
        de_input_dim = class_num
    elif args.task == 'pose':
        de_input_dim = num_joint*2
    elif args.task == 'joint':
        de_input_dim = class_num + num_joint*2
    else:
        raise Exception("Unsupport task!", args.task)

    # ----------------------------------------------------------------------------
    # train loader
    feat_path = os.path.join(args.root_dir, 'group_data', data_name)
    train_set = data.OAD_feat_seq2seq(feat_path, args.mode, args.future_length, 
                    num_joint, pose_flag, pose_1080p_flag, train_flag=True)
    test_set = data.OAD_feat_seq2seq(feat_path, args.mode, args.future_length, 
                    num_joint, pose_flag, pose_1080p_flag, train_flag=False)
    # test loader
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, batch_size=1,
                    shuffle=False, num_workers=args.workers, pin_memory=True)
   
    # --------------------------------------------------------------------------
    # build model
    encoder = lstm.Encoder(input_size, hid_dim, class_num, n_layers=1)
    decoder = lstm.Decoder(de_input_dim, hid_dim, class_num, n_layers=1, 
                            n_pose = num_joint * 2)
    model = lstm.Seq2Seq(encoder, decoder, f_length=args.future_length,
                            task=args.task)
    model = model.cuda()
    #Loss function
    criterion = {0: nn.CrossEntropyLoss().cuda(), 1: nn.SmoothL1Loss().cuda()}
    # Optimizer and lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        
        # traing one epoch
        adjust_learning_rate(optimizer, epoch, args)
        print('Epoch:[{0}/{1}][training stage]'.format(epoch, args.epochs))
        train_accs, train_len_acc, train_loss, train_loss_reg = train(
            train_loader, model, criterion, optimizer, epoch, args)
        
        # testing one epoch
        print('Epoch:[{0}/{1}][validation stage]'.format(epoch, args.epochs))
        test_accs, test_len_acc, test_loss, test_loss_reg = validate(
            test_loader, model, criterion, args)
       

        # save logs every [args.log_freq]
        for k in range(args.future_length):
            if k % args.log_freq == 0:
                writer.add_scalars('Acc@{}'.format(k), 
                        {'train_acc' : train_accs[k].avg,
                        'test_acc' : test_accs[k].avg}, epoch)
        # keep the last logs
        writer.add_scalars('Acc@{}'.format(k), 
                        {'train_acc' : train_accs[k].avg,
                        'test_acc' : test_accs[k].avg}, epoch)
        # write to tensorboard
        writer.add_scalars('Loss', {'train_loss' : train_loss,
                                      'test_loss' : test_loss}, epoch)
        writer.add_scalars('Loss_reg', {'train_loss_reg' : train_loss_reg,
                                      'test_loss_reg' : test_loss_reg}, epoch)
        writer.add_scalars('Len_acc', {'train_len_acc' : train_len_acc,
                                      'test_len_acc' : test_len_acc}, epoch)
        # flush to log
        writer.close()

        # remember best prec@1 and save checkpoint
        test_acc = test_len_acc
        is_best = test_acc > best_prec1
        best_prec1 = max(test_acc, best_prec1)
        save_checkpoint({
            'epoch': epoch,
            'arch': 'seq2seq',
            'state_dict': model.state_dict(),
            # 'state_dict': model.module.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, ckpt_path, best_path)

        print(' *Acc: {acc:.3f}\t Loss: {loss:.4f}\t Acc_best:{best:.3f}'
            .format(acc=test_acc, loss=test_loss, best=best_prec1))
