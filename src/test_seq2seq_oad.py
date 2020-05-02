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
from utils import accuracy, AverageMeter, pred_viz, load_trained_model, str2bool
from utils import SeqCrossEntropyLoss, SeqSmoothL1Loss, f1_score, pose_error, scd

parser = argparse.ArgumentParser(description="Test seq2seq model.")
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--mode', default='flow', type=str, metavar='S', help='rgb, flow, pose')
parser.add_argument('--future_length', default=5, type=int, metavar='N', help='the number of future frames')
parser.add_argument('--pose_flag', default='y', type=str, metavar='S', help='whether to add pose feature')
parser.add_argument('--copy_flag', default='n', type=str, metavar='S', help='whether to add pose feature')


def pose_show(pose, frame_num, k, label, prefix='gt', color='r-'):
    """Show one pose image.
    """
    label_name = {0: 'None', 
                1: 'drinking', 
                2: 'eating', 
                3: 'writing', 
                4: 'opening cupboard', 
                5: 'opening microwave oven', 
                6: 'washing hands', 
                7: 'sweeping', 
                8: 'gargling', 
                9: 'Throwing trash', 
                10: 'wiping'}

    # PARENTS[i] = j indicates that there's a connection between joint i and joint
    # j. The connections form a tree, with the head at the root. See README.txt for
    # joint names.
    PARENTS = [0, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13]
    pose = pose.reshape(-1, 2) * np.array([640, 360])

    # plt.title('#{:04d} +{}, {}:{:d}, {}'.format(
    #     frame_num+1, k, prefix, label, label_name[label]))
    plt.title('#{:04d} +{}'.format(frame_num+1, k))
    plt.xlim(left=0, right=640)
    plt.ylim(bottom=360, top=0)     # set origin at the top left like image
    plt.xticks([])
    plt.yticks([])
    # plt.xaxis.tick_top()            # move x-axis to the top
    # plt.imshow(frame)
    for i in range(len(PARENTS)):
        j = PARENTS[i]
        # Draw line from parent joint (j) to child joint (i)
        # ipdb.set_trace()
        x_locs = pose[[i, j], 0]
        y_locs = pose[[i, j], 1]
        # do not draw bad points (openpose do not detect some joints)
        # ipdb.set_trace()
        if (x_locs==[0, 0]).any() or (y_locs==[0, 0]).any():
            # ipdb.set_trace()
            pass
        else:
            plt.plot(x_locs, y_locs, color)


def pose_save(pose, label, pose_out, label_out, save_dir):
    pose = pose.cpu().numpy()
    label = label.cpu().numpy()
    pose_out = pose_out.cpu().numpy()
    label_out = label_out.cpu().numpy()

    # show pose, default figsize=(6.4, 4.8) inches
    # ipdb.set_trace()
    (f_length, seq_len) = label.shape
    # plt.figure(figsize=(6.4, 1.8 * f_length))
    plt.figure(figsize=(3.2 * f_length, 1.8))

    for frame_num in range(seq_len):
        # example of bad joints
        # if frame_num == 167:
        #     ipdb.set_trace()
        for k in range(f_length):
            pose_pred = pose_out[k, frame_num, :]
            label_pred = label_out[k, frame_num]
            # plt.subplot(f_length, 2, 2*k+1)
            plt.subplot(1, f_length, k+1)
            pose_show(pose_pred, frame_num, k, label_pred, prefix='action', color='b-')

            pose_gt = pose[k, frame_num, :]
            label_gt = label[k, frame_num]
            # plt.subplot(f_length, 2, 2*(k+1))
            pose_show(pose_gt, frame_num, k, label_gt, prefix='action', color='r--')

        plt.show()
        # ipdb.set_trace()
        
        plt.savefig('{}/{}/{:04d}.pdf'.format(save_dir, '', frame_num) )
        plt.clf()
    plt.close()


def test(test_loader, model, criterion, log_dir, f_length, copy_flag):
    """Test trained model on testing data.
    """
    losses_reg = AverageMeter('Loss', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    top1 = [AverageMeter('Acc@{}'.format(k), ':6.2f') for k in range(f_length)]
    len_acc = AverageMeter('Len_acc', ':6.2f')
    len_f1 = AverageMeter('Len_f1', ':.4e')
    len_pose = AverageMeter('Len_pose', ':.4e')
    len_scd = AverageMeter('Len_pose', ':.4e')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (data, label, pose) in enumerate(test_loader):
            # ipdb.set_trace()
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
            
            loss = 0
            loss_step = 0
            length_acc = 0
            length_f1 = 0
            length_scd = 0
            length_pose = 0
            for k in range(f_length):

                # decoder baseline: copy the first recognition
                if copy_flag:
                    class_out[k] = class_out[0]
                    pose_out[k] = pose_out[1]

                # measure accuracy 
                acc = accuracy(class_out[k], label[0, k, :])
                top1[k].update(acc[0].item(), label.size(2))
                length_acc += acc[0].item()
                # f1 score
                f1_res = f1_score(class_out[k], label[0, k, :])
                length_f1 += f1_res[2]
                # scd
                scd_pred = scd(class_out[k], label[0, k, :])
                length_scd += scd_pred
                # pose error
                pose_err = pose_error(pose_out[k].cpu().data.numpy(), \
                            pose[0, k, :].cpu().data.numpy(), metric='L2')
                length_pose += pose_err
                
            
            # average all predited timesteps
            avg_acc = length_acc/f_length
            len_acc.update(avg_acc, label.size(2))
            # f1 score for a sequence
            avg_f1 = length_f1/f_length
            len_f1.update(avg_f1, 1)
            # scd for a sequence
            avg_scd = length_scd / f_length
            len_scd.update(avg_scd, 1)
            # pose error
            avg_pose = length_pose / f_length
            len_pose.update(avg_pose, label.size(2))

            # print accuracy of each sequence
            print((' Sequence {0}:\t avg_acc: {acc:.3f}\t avg_f1: {f1:.4f}\t ' 
                'avg_scd: {scd:.4f}\t avg_pose: {pose:.3f}').format(i+1, \
                acc=avg_acc, f1=avg_f1, scd=avg_scd, pose=avg_pose))

            # ------------------------------------------------------------------
            # # show pose iamge
            # # uncomment if you want to save the results.
            # save_dir = os.path.join(log_dir, 'show/', str(i+1))
            # if not os.path.isdir(save_dir):
            #     os.makedirs(save_dir)
            # label_out = torch.max(class_out, 2)[1]
            # n_class = class_out.shape[2]
            # pose_save(pose.squeeze(), label.squeeze(), pose_out, label_out, save_dir)

            # # visualize the predicted results
            # # ipdb.set_trace()
            # if copy_flag:
            #     save_name = 'copy.pdf'
            # else:
            #     save_name = 'decoder.pdf'
            # pred_viz(label[0].cpu(), label_out.cpu(), avg_acc, n_class, save_dir,
            #          save_name)
    return len_acc.avg, len_f1.avg, len_scd.avg, len_pose.avg


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    root_dir = '../dataset_public/OAD/'
    model_folder = 'trained_models'
    model_exp = 'seq2seq'
    log_folder = 'logs'
    best_prec1 = 0
    class_num = 11
    hid_dim = 128
    feat_size = 512
    num_joint = 15
    pose_size = 2 * num_joint * args.future_length
    data_name = 'feats.h5'
    copy_flag = str2bool(args.copy_flag)
    pose_1080p_flag = True

    # --------------------------------------------------------------------------
    # path config
    exp_name = '{}_future{}'.format(model_exp, args.future_length)
    log_name = "{}_{}_pose{}".format(exp_name, args.mode, str2bool(args.pose_flag))
    log_dir = os.path.join(root_dir, log_folder, exp_name, log_name)

    model_dir = os.path.join(root_dir, model_folder, exp_name, log_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    ckpt_path = os.path.join(model_dir, 'checkpoint.pth.tar')
    best_path = os.path.join(model_dir, 'model_best.pth.tar')
    
    # check if the experiment has been performed
    if not os.path.isdir(log_dir):
        print('Cannot find {}. Please train a model before testing.'.format(model_dir))
        sys.exit(0)
    
    
    # --------------------------------------------------------------------------
    # mode config and dataloader
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

    # load all data
    feat_path = os.path.join(root_dir, 'group_data', data_name)
    test_set = data.OAD_feat_seq2seq(feat_path, args.mode, args.future_length, 
                    num_joint, pose_flag, pose_1080p_flag, train_flag=False)
    test_loader = DataLoader(dataset=test_set, batch_size=1,
                    shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # --------------------------------------------------------------------------
    # build model
    encoder = lstm.Encoder(input_size, hid_dim, class_num, n_layers=1)
    decoder = lstm.Decoder(class_num, hid_dim, class_num, n_layers=1, 
                            n_pose = num_joint * 2)
    model = lstm.Seq2Seq(encoder, decoder, f_length=args.future_length)
    model = load_trained_model(model, ckpt_path)
    model = model.cuda()
    
    #Loss function
    criterion = {0: nn.CrossEntropyLoss().cuda(), 
                 1: nn.SmoothL1Loss().cuda()}
    cudnn.benchmark = True
    
    # --------------------------------------------------------------------------
    # test
    len_acc, test_f1, test_scd, test_pose= test(test_loader, model, criterion, 
                        log_dir, args.future_length, copy_flag)
    print((" *Len_acc: {acc:.3f}\t F1_score: {f1:.4f} \t SCD: {scd:.4f} \t " 
            "Pose_err * 100: {pe:.4f}").format(acc=len_acc, f1=test_f1, 
            scd=test_scd, pe=test_pose*100))
