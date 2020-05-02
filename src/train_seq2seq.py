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
parser.add_argument('--pw', default=100, type=float, metavar='M', help='the weight of pose regression loss.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')


parser.add_argument('--root_dir', default='../data/ikea-fa-release-data', type=str, metavar='S', help='dataset directory')
parser.add_argument('--model_folder', default='trained_models', type=str, metavar='S', help='model folder')
parser.add_argument('--model_exp', default='seq2seq', type=str, metavar='S', help='model experiments')
parser.add_argument('--log_folder', default='logs', type=str, metavar='S', help='logs for tensorboard')
parser.add_argument('--feat_file', default='feats/feats_len5.h5', type=str, metavar='S', help='saved feature file')
parser.add_argument('--train_rec', default='train_30fps.csv', type=str, metavar='S', help='training split')
parser.add_argument('--test_rec', default='test_30fps.csv', type=str, metavar='S', help='testing split')
parser.add_argument('--print_freq', default=20, type=int, metavar='N', help='print frequency (default: 20)')
parser.add_argument('--log_freq', default=5, type=int, metavar='N', help='tensorboard log frequency (default: 5)')

parser.add_argument('--pose_flag', default='n', type=str, metavar='S', help='whether to add pose feature')
parser.add_argument('--task', default='joint', type=str, metavar='S', help="task: 'class', 'pose', 'joint'")
parser.add_argument('--phase', default='rgb', type=str, metavar='S', help='rgb, flow, pose')
parser.add_argument('--future_length', default=0, type=int, metavar='N', help='the number of future frames')
parser.add_argument('--fix', default='n', type=str, metavar='S', help='whether to fix weights')

def main():
    # ----------------------------------------------------------------------------
    # configs
    args = parser.parse_args()
    print(args)
    best_prec1 = 0
    class_num = 13
    num_joint = 8
    hidden_size = 128

    # log dir
    log_name = "{}_pose{}_future{}_seq2seq_{}".format(args.phase, 
                str2bool(args.pose_flag), args.future_length, args.task)
    log_dir = os.path.join(args.root_dir, args.log_folder, args.model_exp, log_name)
    # model dir
    model_dir = os.path.join(args.root_dir, args.model_folder, args.model_exp)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    ckpt_path = os.path.join(model_dir, log_name + '_checkpoint.pth.tar')
    best_path = os.path.join(model_dir, log_name + '_model_best.pth.tar')
    
    # encoder input config according to args.phase
    pose_flag = str2bool(args.pose_flag)
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
    train_recs = parse_rec(args.root_dir, args.train_rec)
    train_set = data.Ikea_feat_seq2seq(root = args.root_dir, rec = train_recs, future_length=args.future_length,
                    phase = args.phase, pose_flag = pose_flag, feat_file = args.feat_file)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, pin_memory=True)
    # test loader
    test_recs = parse_rec(args.root_dir, args.test_rec)
    test_set = data.Ikea_feat_seq2seq(root = args.root_dir, rec = test_recs, future_length=args.future_length,
                    phase = args.phase, pose_flag = pose_flag, feat_file = args.feat_file)
    test_loader = DataLoader(dataset=test_set, batch_size=1,
                    shuffle=False, num_workers=args.workers, pin_memory=True)


    # ----------------------------------------------------------------------------
    # build model
    # model = lstm.LSTMmultiLabel(input_size, hidden_size, class_num, n_layers=1, f_length=args.future_length)
    encoder = lstm.Encoder(input_size, hidden_size, class_num, n_layers=1)
    decoder = lstm.Decoder(de_input_dim, hidden_size, class_num, n_layers=1, 
                            n_pose = num_joint*2)
    model = lstm.Seq2Seq(encoder, decoder, f_length=args.future_length, 
                            task=args.task)
    model = model.cuda()
    #Loss function
    criterion = {0: nn.CrossEntropyLoss().cuda(), 1: nn.SmoothL1Loss().cuda()}
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
    
    for epoch in range(1, args.epochs+1):
        
        # traing one epoch on all test_loaders
        adjust_learning_rate(optimizer, epoch, args)
        print('Epoch:[{0}/{1}][training stage]'.format(epoch, args.epochs))
        train_accs, train_len_acc, train_loss, train_loss_reg = train(
            train_loader, model, criterion, optimizer, epoch, args)
        
        # testing one epoch on all test_loaders
        print('Epoch:[{0}/{1}][validation stage]'.format(epoch, args.epochs))
        test_accs, test_len_acc, test_loss, test_loss_reg = validate(
            test_loader, model, criterion, epoch, args)
       
        
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
            'epoch'     : epoch,
            'arch'      : 'lstm',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, ckpt_path, best_path)
        
        print(' *Acc: {acc:.3f}\t Loss: {loss:.4f}\t Acc_best:{best:.3f}'
            .format(acc=test_acc, loss=test_loss, best=best_prec1))
    
    # add model as graph tensorboard
    # writer.add_graph(model)
    # writer.close()


def train(train_loader, model, criterion, optimizer, epoch, args):
    
    losses = AverageMeter('Loss', ':.4e')
    losses_reg = AverageMeter('Loss', ':.4e')
    top1 = [AverageMeter('Acc@{}'.format(k), ':6.2f') for k in range(args.future_length)]
    len_acc = AverageMeter('Len_acc', ':6.2f')
    
    #switch to train mode
    model.train()
    for i, (data, label, pose) in enumerate(tqdm(train_loader)):
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

            # if args.task == 'class':
            # elif args.task == 'pose':
            # elif args.task == 'joint':
            # else:
            #     raise Exception("Unsupport task!", args.task)
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


def validate(test_loader, model, criterion, epoch, args):
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

                if args.task == 'class':
                    loss += loss_cls
                elif args.task == 'pose':
                    loss += loss_reg
                elif args.task == 'joint':
                    loss += (loss_cls + args.pw * loss_reg)
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
	main()
