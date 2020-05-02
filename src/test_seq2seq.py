import os, sys, time, shutil, argparse, ipdb
import numpy as np
from tqdm import tqdm
# enable plt on server
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

import data, video_transforms
from nets import resnet, lstm
from utils import accuracy, parse_rec, AverageMeter, pred_viz
from utils import SeqCrossEntropyLoss, SeqSmoothL1Loss


def get_input_size(phase, pose_flag):
    """Determine the input size of LSTM according to 'phase' and 'pose_flag'.
    """
    if phase == 'rgb' or phase == 'flow':
        input_size = 512 
    elif phase == 'rgb_flow':
        input_size = 512 * 2 
    elif phase == 'pose':
        input_size = 80  
        pose_flag = False
    else:
        print("Only support phase = 'rgb', 'flow' or 'pose' ")
    # ipdb.set_trace()
    if pose_flag:
        # resnet feature + coordinates of the first 8 joints
        input_size += 80 
    return input_size


def load_trained_model(model, ckpt_path):
    """ Load trained model.
    """
    if os.path.isfile(ckpt_path):
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print(("=> best_prec1: '{}' (epoch {})"
              .format(best_prec1, start_epoch)))
    else:
        print("=> no checkpoint found.")
    
    return model


def pose_show(pose, frame_num, k, label, prefix='gt', color='r-'):
    """Show one pose image.
    """
    label_name = {0: 'None', 
                1: 'attach leg 1', 
                2: 'attach leg 2', 
                3: 'attach leg 3', 
                4: 'attach leg 4', 
                5: 'detach leg 1', 
                6: 'detach leg 2', 
                7: 'detach leg 3', 
                8: 'detach leg 4', 
                9: 'flip over', 
                10: 'spin in', 
                11: 'spin out', 
                12: 'pick leg'}
    # PARENTS[i] = j indicates that there's a connection between joint i and joint
    # j. The connections form a tree, with the head at the root. See README.txt for
    # joint names.
    PARENTS = [0, 0, 1, 2, 3, 1, 5, 6]

    # normalize pose and resahpe to [x_0, y_0, x_1, y_1, ..., x_7, y_7]
    # for kk in range(self.frame_length):
    #     norm_pose = (pose[kk][:8] / np.array([1920, 1080])[None, :]).astype(np.float32)
    #     clip_pose[16*kk : 16*(kk+1)] = norm_pose.reshape(16)[:]
    # ipdb.set_trace()
    pose = pose.reshape(-1, 2) * np.array([640, 360])

    
    plt.title('#{:04d} +{}, {}:{:d}, {}'.format(
        frame_num+1, k, prefix, label, label_name[label]))
    plt.xlim(left=0, right=640)
    plt.ylim(bottom=360, top=0)     # set origin at the top left like image
    plt.xticks([])
    plt.yticks([])
    # plt.xaxis.tick_top()            # move x-axis to the top
    # plt.imshow(frame)
    for i in range(len(PARENTS)):
        j = PARENTS[i]
        # if j:     # remove head joint
        # Draw line from parent joint (j) to child joint (i)
        x_locs = pose[[i, j], 0]
        y_locs = pose[[i, j], 1]
        plt.plot(x_locs, y_locs, color)


def pose_save(pose, label, pose_out, label_out, save_dir):
    pose = pose.cpu().numpy()
    label = label.cpu().numpy()
    pose_out = pose_out.cpu().numpy()
    label_out = label_out.cpu().numpy()

    # show pose, default figsize=(6.4, 4.8) inches
    # ipdb.set_trace()
    (f_length, seq_len) = label.shape
    plt.figure(figsize=(6.4, 1.8 * f_length))

    for frame_num in range(seq_len):
        
        for k in range(f_length):
            pose_pred = pose_out[k, frame_num, :]
            label_pred = label_out[k, frame_num]
            plt.subplot(f_length, 2, 2*k+1)
            pose_show(pose_pred, frame_num, k, label_pred, prefix='pred', color='b-')


            pose_gt = pose[k, frame_num, :]
            label_gt = label[k, frame_num]
            plt.subplot(f_length, 2, 2*(k+1))
            pose_show(pose_gt, frame_num, k, label_gt, prefix='gt', color='r-')

        plt.show()
        ipdb.set_trace()
        
        plt.savefig('{}/{:04d}.png'.format(save_dir, frame_num) )
        plt.clf()
    plt.close()


def test(test_loader, model, criterion, log_dir, f_length, copy_flag):
    """Test trained model on testing data.
    """
    losses_reg = AverageMeter('Loss', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    top1 = [AverageMeter('Acc@{}'.format(k), ':6.2f') for k in range(f_length)]
    len_acc = AverageMeter('Len_acc', ':6.2f')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (data, label, pose) in enumerate(test_loader):
            # label.shape = (batch, f_length, seq_len), here batch = 1
            # data.shape = (batch, seq_len, feat_dim)
            # poae.shape = (batch, f_length, seq_len, 16)
            label = label.long().cuda(non_blocking=True)
            data = data.cuda(non_blocking=True)
            pose = pose.cuda(non_blocking=True)
        
            # class_out.shape = (f_length, seq_len, n_class)
            # pose_out.shape = (f_length, seq_len, n_pose)
            # label_out.shape = (f_length, seq_len)
            # ipdb.set_trace()
            class_out, pose_out = model(data)
            label_out = torch.max(class_out, 2)[1]
            n_class = class_out.shape[2]
            
            loss = 0
            loss_step = 0
            length_acc = 0
            for k in range(f_length):

                # decoder baseline: copy the first recognition
                if copy_flag:
                    class_out[k] = class_out[0]
                
                # compute loss
                # loss += criterion[0](output[k], label[0, k, :])
                loss_cls = criterion[0](class_out[k], label[0, k, :])
                loss_reg = criterion[1](pose_out[k], pose[0, k, :])
                loss += (loss_cls + 100 * loss_reg)
                loss_step += loss_reg

                # measure accuracy 
                acc = accuracy(class_out[k], label[0, k, :], topk=(1, ))
                top1[k].update(acc[0].item(), label.size(2))
                length_acc += acc[0].item()
                
            losses.update(loss.item()/f_length, label.size(2))
            losses_reg.update(loss_step.item()/f_length, label.size(2))
            # average all predited timesteps
            avg_acc = length_acc/f_length
            len_acc.update(avg_acc, label.size(2))

            # print accuracy of each sequence
            print(' Sequence {0}:\t Len_acc: {acc:.3f}\t Loss_reg: {loss:.4f} '
                .format(i+1, acc=avg_acc, loss=losses_reg.avg))

            # # show pose iamge
            # save_dir = os.path.join(log_dir, 'show', str(i+1))
            # if not os.path.isdir(save_dir):
            #     os.makedirs(save_dir)
            # # pose_save(pose.squeeze(), label.squeeze(), pose_out, label_out, save_dir)

            # # visualize the predicted results
            # # ipdb.set_trace()
            # if copy_flag:
            #     save_name = 'copy.png'
            # else:
            #     save_name = 'decoder.png'
            # pred_viz(label[0].cpu(), label_out.cpu(), avg_acc, n_class, save_dir, save_name)

            
    return len_acc.avg, losses.avg, losses_reg.avg


if __name__ == '__main__':
    root_dir = '../dataset_public/ikea-fa-release-data'
    model_folder = 'trained_models'
    feat_file = 'data/feats_len5.h5'
    test_rec = 'test_30fps.csv'
    class_num = 13
    hidden_size = 128
    f_length = 10

    phase = 'pose'
    log_folder = 'logs/seq2seq'
    pose_flag = True
    copy_flag = False

    input_size = get_input_size(phase, pose_flag)
    log_name = "{}_pose{}_future{}_seq2seq".format(phase, pose_flag, f_length)
    model_dir = os.path.join(root_dir, model_folder, log_folder[5:])
    ckpt_path = os.path.join(model_dir, log_name + '_checkpoint.pth.tar')
    best_path = os.path.join(model_dir, log_name + '_model_best.pth.tar')
    log_dir = os.path.join(root_dir, log_folder, log_name)
    if not os.path.isdir(log_dir):
        print('Cannot find {}. Please train a model before testing.'.format(log_dir))
        sys.exit()
    
    
    # testing loader
    test_recs = parse_rec(root_dir, test_rec)
    test_set = data.Ikea_feat_seq2seq(root = root_dir, rec = test_recs, future_length=f_length,
                    phase = phase, pose_flag = pose_flag, feat_file = feat_file)
    test_loader = DataLoader(dataset=test_set, batch_size=1,
                    shuffle=False, num_workers=4, pin_memory=True)

    # lstm model to be trained
    encoder = lstm.Encoder(input_size, hidden_size, class_num, n_layers=1, f_length=f_length)
    decoder = lstm.Decoder(class_num, hidden_size, class_num, n_layers=1, f_length=f_length)
    model = lstm.Seq2Seq(encoder, decoder, f_length=f_length)
    model = load_trained_model(model, ckpt_path)
    model = model.cuda()
    
    #Loss function
    criterion = {0: nn.CrossEntropyLoss().cuda(), 
                 1: nn.SmoothL1Loss().cuda()}
    cudnn.benchmark = True
    
    len_acc, test_loss, test_loss_reg = test(test_loader, model, criterion, log_dir, f_length, copy_flag)
    print(' *Len_ccc: {acc:.3f}\t Loss_reg: {loss:.4f}'.format(acc=len_acc, loss=test_loss_reg))
