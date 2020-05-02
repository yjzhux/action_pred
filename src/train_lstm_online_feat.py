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
from utils import accuracy, save_checkpoint, adjust_learning_rate, parse_rec, str2bool
from utils import AverageMeter, FeatureExtractor

parser = argparse.ArgumentParser(description="PyTorch Spatial ")
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_steps', default=20, type=int, metavar='LRSteps', help='epochs to decay lr by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 20)')
parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--phase', default='rgb', type=str, metavar='S', help='rgb, flow, pose')
parser.add_argument('--frame_length', default=5, type=int, metavar='N', help='the number of stacked frames (default: 5)')
parser.add_argument('--root_dir', default='../dataset_public/ikea-fa-release-data', type=str, metavar='S', help='dataset directory')
parser.add_argument('--model_folder', default='trained_models', type=str, metavar='S', help='model folder')
parser.add_argument('--log_folder', default='logs', type=str, metavar='S', help='logs for tensorboard')
parser.add_argument('--train_rec', default='train_30fps.csv', type=str, metavar='S', help='training split')
parser.add_argument('--test_rec', default='test_30fps.csv', type=str, metavar='S', help='testing split')
parser.add_argument('--weighted_loss', default='n', type=str, metavar='S', help='whether to use weighted loss')
parser.add_argument('--fix', default='n', type=str, metavar='S', help='whether to fix weights')
parser.add_argument('--pose', default='n', type=str, metavar='S', help='whether to use pose')

def main():
    args = parser.parse_args()
    print(args)
    best_prec1 = 0
    class_num = 13
    log_name = "{}_pose{}_len{}_lstm".format(args.phase, str2bool(args.pose), args.frame_length)
    ckpt_path = os.path.join(args.root_dir, args.model_folder, log_name + '_checkpoint.pth.tar')
    best_path = os.path.join(args.root_dir, args.model_folder, log_name + '_model_best.pth.tar')
    log_dir = os.path.join(args.root_dir, args.log_folder, log_name)
    hidden_size = 128
    if str2bool(args.pose):
        input_size = 512 + 8*2*args.frame_length   # resnet feature + coordinates of the first 8 joints
    else:
        input_size = 512
    # ipdb.set_trace()
    if str2bool(args.weighted_loss):
        class_weights = [9, 76, 57, 90, 52, 138, 128, 110, 95, 13, 2, 3, 24]
    else:
        class_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # ipdb.set_trace()
    if args.phase == 'rgb':
        # scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406] * args.frame_length
        clip_std = [0.229, 0.224, 0.225] * args.frame_length
        in_channel = 3 * args.frame_length
        model_best = os.path.join(args.root_dir, args.model_folder, 'rgb_len5_model_best.pth.tar')

    elif args.phase == 'flow':
        # scale_ratios = [1.0, 0.875, 0.75]
        clip_mean = [0.5, 0.5] * args.frame_length
        clip_std = [0.226, 0.226] * args.frame_length
        in_channel = 2 * args.frame_length
        model_best = os.path.join(args.root_dir, args.model_folder, 'flow_len5_checkpoint.pth.tar')
    
    elif args.phase == 'gray':
        clip_mean = [0.5] * args.frame_length
        clip_std = [0.226] * args.frame_length
        in_channel = 1 * args.frame_length
    else:
        print('Only rgb, fow and gray supported.')

    # transform setting
    transform=video_transforms.Compose([
                video_transforms.ToTensor(),
                video_transforms.Normalize(mean=clip_mean,std=clip_std)
                ])

    train_recs = parse_rec(args.root_dir, args.train_rec)
    test_recs = parse_rec(args.root_dir, args.test_rec)

    # -----------------------------------------------------------------------------
    # feature extracting model
    feat_model = resnet.resnet18(pretrained=False, in_channel=in_channel, 
                            num_classes=class_num, fix=str2bool(args.fix))
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
    
    
    # lstm model to be trained
    model = lstm.LSTM(input_size, hidden_size, class_num, n_layers=1)
    model = model.cuda()
    #Loss function
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights)).cuda()
    # Optimizer and lr_scheduler
    if str2bool(args.fix):
        optim_para = filter(lambda p:p.requires_grad, model.parameters())
    else:
        optim_para = model.parameters()
    optimizer = torch.optim.Adam(optim_para, lr=args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True
    
    # ----------------------------------------------------------------------------
    #Training & testing
    # use tensorboard to visualize
    # ipdb.set_trace()
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
            train_set = data.Ikea_seq(root = args.root_dir, rec = train_recs[idx],
                        phase = args.phase, frame_length = args.frame_length,
                        transform = transform)
            train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)
            
            feats, labels = extract_feat(train_loader, feat_model, input_size)

            acc, loss = train(feats, labels, model, criterion, optimizer, epoch, args)
            train_acc.update(acc)
            train_loss.update(loss)
            print(' Training on {rec}\t Acc: {acc:.3f}\t Loss: {loss:.4f} '
                .format(rec=train_recs[idx], acc=acc, loss=loss))

        # testing one epoch on all test_loaders
        print('Epoch:[{0}/{1}][validation stage]'.format(epoch, args.epochs))
        for idx in range(len(test_recs)):
            test_set = data.Ikea_seq(root = args.root_dir, rec = test_recs[idx],
                        phase = args.phase, frame_length = args.frame_length,
                        transform = transform)
            test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)
            feats, labels = extract_feat(test_loader, feat_model, input_size)
            acc, loss = validate(feats, labels, model, criterion, epoch, args)
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
            'arch': 'lstm',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, ckpt_path, best_path)



def extract_feat(data_loader, model, feat_size):
    '''Extract deep features and concatenate pose features.
    '''
    exact_list=["avgpool"]  # features before dropout
    extractor=FeatureExtractor(model, exact_list)
    feats = torch.zeros([1, len(data_loader), feat_size], dtype=torch.float)
    labels = torch.zeros(len(data_loader), dtype=torch.int8)

    with torch.no_grad():
        for i, (data, label, pose) in enumerate(data_loader):
            # ipdb.set_trace()
            data = data.cuda(non_blocking=True)
            # pose = pose.cuda(non_blocking=True)
            feat = extractor(data)[0]
            feat = feat.view(feat.size(0), -1)  # feat.shape is (1, 512)
            if feat_size > feat.shape[1]:
                feats[0, i, :] = torch.cat((feat.cpu(), pose), 1).squeeze()
            else:
                feats[0, i, :] = feat.cpu().squeeze()[:]

            labels[i] = label
    return feats, labels


def train(data, label, model, criterion, optimizer, epoch, args):
    
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    #switch to train mode
    model.train()
    label = label.long().cuda(non_blocking=True)
    data = data.cuda(non_blocking=True)
    
    # compute output
    # ipdb.set_trace()
    output, _ = model(data)
    loss = criterion(output.squeeze(), label)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.squeeze(), label, topk=(1, 5))
    losses.update(loss.item(), data.size(1))
    top1.update(prec1[0], data.size(1))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return top1.avg, losses.avg


def validate(data, label, model, criterion, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        
        label = label.long().cuda(non_blocking=True)
        data = data.cuda(non_blocking=True)

        # compute output
        output, _ = model(data)
        loss = criterion(output.squeeze(), label)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.squeeze(), label, topk=(1, 5))
        losses.update(loss.item(), data.size(1))
        top1.update(prec1[0], data.size(1))

    return top1.avg, losses.avg


if __name__ == '__main__':
	main()
