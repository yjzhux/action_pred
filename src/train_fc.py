import distutils.util
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


def str2bool(v):
    return bool(distutils.util.strtobool(v))


parser = argparse.ArgumentParser(description="PyTorch Spatial ")
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_steps', default=20, type=int, metavar='LRSteps', help='epochs to decay lr by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')

parser.add_argument('--root_dir', default='../data/ikea-fa-release-data', type=str, metavar='S', help='dataset directory')
parser.add_argument('--model_folder', default='trained_models', type=str, metavar='S', help='model folder')
parser.add_argument('--log_folder', default='logs', type=str, metavar='S', help='logs for tensorboard')
parser.add_argument('--model_exp', default='fc', type=str, metavar='S', help='model experiments')

parser.add_argument('--train_rec', default='train_30fps.csv', type=str, metavar='S', help='training split')
parser.add_argument('--test_rec', default='test_30fps.csv', type=str, metavar='S', help='testing split')
parser.add_argument('--weighted_loss', default=False, type=bool, metavar='B', help='whether to use weighted loss')
parser.add_argument('--fix', default=False, type=str2bool, metavar='B', help='whether to fix weights')

# parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 20)')
parser.add_argument('--phase', default='rgb', type=str, metavar='S', help='rgb, flow, pose')
parser.add_argument('--frame_length', default=5, type=int, metavar='N', help='the number of stacked frames (default: 5)')
parser.add_argument('--dropout', default=0.7, type=float, metavar='D', help='probability for dropout layer')

def main():
    args = parser.parse_args()
    best_prec1 = 0
    class_num = 13
    
    # Use appropriate default, e.g., 'rgb_len5', 'flow_len5'
    exp_name = '%s_len%d' % (args.phase, args.frame_length)
    log_dir = os.path.join(args.root_dir, args.log_folder, args.model_exp, exp_name)
    # model path
    model_dir = os.path.join(args.root_dir, args.model_folder, args.model_exp)
    if os.path.isdir(model_dir):
        print("Exp: '", model_dir,  "' already exists.")
        print('Please manually delete if you want to re-run it.')
    else:
        os.makedirs(model_dir)
        print("Exp: '", model_dir, "' created.")
    ckpt_path = os.path.join(model_dir, exp_name + '_checkpoint.pth.tar')
    best_path = os.path.join(model_dir, exp_name + '_model_best.pth.tar')
    

    if args.weighted_loss:
        class_weights = [9, 76, 57, 90, 52, 138, 128, 110, 95, 13, 2, 3, 24]
    else:
        class_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    if args.phase == 'rgb':
        # scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406] * args.frame_length
        clip_std = [0.229, 0.224, 0.225] * args.frame_length
        in_channel = (args.frame_length, 3)

    elif args.phase == 'flow':
        # scale_ratios = [1.0, 0.875, 0.75]
        clip_mean = [0.5, 0.5] * args.frame_length
        clip_std = [0.226, 0.226] * args.frame_length
        in_channel = (args.frame_length, 2)
    elif args.phase == 'gray':
        clip_mean = [0.5] * args.frame_length
        clip_std = [0.226] * args.frame_length
        in_channel = (args.frame_length, 1)
    else:
        print('Only rgb, fow and gray supported.')

    train_recs = parse_rec(args.root_dir, args.train_rec)
    test_recs = parse_rec(args.root_dir, args.test_rec)

    # -----------------------------------------------------------------------------
    # Build k channel network with pre-trained weight
    model = channel_resnet18(pretrained=True, in_channel=in_channel,
                             num_classes=class_num, fix=args.fix,
                             dropout=args.dropout)
    #Replace fc1000 with fc101
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, class_num)
    #convert model to gpu
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    #Loss function
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights)).cuda()
    # Optimizer and lr_scheduler
    if args.fix:
        optim_para = filter(lambda p:p.requires_grad, model.parameters())
    else:
        optim_para = model.parameters()
    optimizer = torch.optim.SGD(optim_para, momentum=args.momentum,
                                lr=args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True
    
    # ----------------------------------------------------------------------------
    #Training & testing
    # Make sure that the model folder exists
    os.makedirs(args.model_folder, exist_ok=True)
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
        shffle_num = np.random.permutation(len(train_recs))
        for idx in tqdm(shffle_num):
            train_set = data.Ikea_single(root = args.root_dir, rec = train_recs[idx],
                        phase = args.phase, frame_length = args.frame_length,
                        clip_mean=clip_mean, clip_std=clip_std)
            train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers, pin_memory=True)
            acc, loss = train(train_loader, model, criterion, optimizer, epoch, args)
            train_acc.update(acc)
            train_loss.update(loss)
            print(' Training on {rec}\t Acc: {acc:.3f}\t Loss: {loss:.4f} '
                .format(rec=train_recs[idx], acc=acc, loss=loss))

        # testing one epoch on all test_loaders
        print('Epoch:[{0}/{1}][validation stage]'.format(epoch, args.epochs))
        for idx in range(len(test_recs)):
            test_set = data.Ikea_single(root = args.root_dir, rec = test_recs[idx],
                        phase = args.phase, frame_length = args.frame_length,
                        clip_mean=clip_mean, clip_std=clip_std)
            test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)
            acc, loss = validate(test_loader, model, criterion, epoch, args)
            test_acc.update(acc)
            test_loss.update(loss)
            print(' Testing on {rec}\t Acc: {acc:.3f}\t Loss: {loss:.4f} '
                .format(rec=test_recs[idx], acc=acc, loss=loss))
       

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
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, ckpt_path, best_path)


def train(train_loader, model, criterion, optimizer, epoch, args):
    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    #switch to train mode
    model.train()
    
    # progress = tqdm(train_loader)
    for i, (data,label) in enumerate(train_loader):
        
        label = label.cuda(non_blocking=True)
        data = data.cuda(non_blocking=True)
        # ipdb.set_trace()

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
        
    #show results
    # print('Epoch: [{0}]\t'
    #         #   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #         #   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
    #             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
    #             'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
    #             epoch, loss=losses, top1=top1, top5=top5))
    return top1.avg, losses.avg


def validate(test_loader, model, criterion, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

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
        top5.update(prec5[0], data.size(0))

        # ii = i+1
        # if ii % args.print_freq == 0:
        #     # Show info on console
        #     print('Epoch: [{0}], Testing[{1}/{2}]\t'
        #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch,
        #                i, len(test_loader), batch_time=batch_time, loss=losses,
        #                top1=top1, top5=top5))    

    # print(' * Prec@1: {top1.avg:.3f} Prec@5: {top5.avg:.3f} Loss: {loss.avg:.4f} '.format(top1=top1, top5=top5, loss=losses))
    return top1.avg, losses.avg


if __name__ == '__main__':
	main()
