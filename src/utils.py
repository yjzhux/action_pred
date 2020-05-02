import os, csv, shutil, argparse, ipdb
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

# enable plt on server
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pylab as plt

# ------------------------------------------------------------------------------
# evaluation metrics
def pose_error(pose_out, pose_gt, metric='L2'):
    err = 0.0
    seq_len, num_coor = pose_gt.shape
    for idx in range(seq_len):
        out = pose_out[idx]
        gt = pose_gt[idx]
        # Do NOT take into account these bad joints when measuring error.
        out = out[gt > 0].reshape((-1, 2))
        gt = gt[gt > 0].reshape((-1, 2))
        # ipdb.set_trace()
        if metric == 'L2':
            err_matrix = euclidean_distances(out, gt)
        elif metric == 'L1':
            err_matrix = manhattan_distances(out, gt)
        
        err += err_matrix.trace() / (num_coor/2)
    
    return err / seq_len


def scd(output, target, topk=(1,)):
    '''The Scaled Chaning Distance (SCD) between the predicted distance and ground truth.
    '''
    scd = 0.0
    gt_num = 0
    pred_num = 0

    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    preds = pred.squeeze().cpu().data.numpy()
    labels = target.cpu().data.numpy()
    # ipdb.set_trace()

    for idx in range(len(labels) - 1):
        if labels[idx] != labels[idx+1]:
            gt_num += 1
        if preds[idx] != preds[idx+1]:
            pred_num += 1
    
    return float(pred_num) / gt_num


def f1_score(output, target, topk=(1,), avg = 'macro'):
    """Computes the f1 score over the k top predictions for the specified values
    of k. There is also another function specificlly for f1_score in sklearn, 
    which is called sklearn.metrics.f1_score().
    https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/metrics/classification.py#L950
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    y_true = target.cpu().data.numpy()
    y_pred = pred[0].cpu().data.numpy()
    res = precision_recall_fscore_support(y_true, y_pred, average=avg)

    return res


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def seq_accuracy(seq_output, seq_target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k.
    seq_target.shape = (batch_size, seq_length, 1)
    """
    # concatenate all batches to one long sequence 
    output = seq_output[0][:]
    target = seq_target[0][:] 
    for idx in range(1, seq_target.size(0)):
        output = torch.cat((output, seq_output[idx][:]), 0)
        target = torch.cat((output, seq_target[idx][:]), 0)

    res = accuracy(output, target, topk)
    return res


class SeqCrossEntropyLoss(nn.Module):
    """SeqCrossEntropyLoss"""
    def __init__(self):
        super(SeqCrossEntropyLoss, self).__init__()

    def forward(self, output, y):
        batch_size = y.size(0)
        loss_func = nn.CrossEntropyLoss()

        # Loss from one sequence
        # output will be batch_size*seq_len*num_classes an y will be batch_size*seq_len*1
        loss = loss_func(output[0, :, :].contiguous(), y[0, :].contiguous().view(-1)) 
        for i in range(1, batch_size):
            loss += loss_func(output[i, :, :].contiguous(), y[i, :].contiguous().view(-1))

        return loss


class SeqSmoothL1Loss(nn.Module):
    """ SeqSmoothL1Loss"""
    def __init__(self):
        super(SeqSmoothL1Loss, self).__init__()

    def forward(self, output, y):
        batch_size = y.size(0)
        loss_func = nn.SmoothL1Loss()

        # Loss from one sequence
        # output will be batch_size*seq_len*n_pose an y will be batch_size*seq_len*n_pose
        loss = loss_func(output[0, :, :].contiguous(), y[0, :,:].contiguous()) 
        for i in range(1, batch_size):
            loss += loss_func(output[i, :, :].contiguous(), y[i, :, :].contiguous())

        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


# ------------------------------------------------------------------------------
# model related
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every lr_steps epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_steps))
    print('learning rate: ',lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, ckpt_path, best_path):
    '''Save current checkpoint as well as the best model.'''
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copyfile(ckpt_path, best_path)


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


class FeatureExtractor(nn.Module):
    '''Extract the features from any layers of a pretrained model.
    FeatureExtractor only works for networks with submodules that are the layers. 
    The problem is not solved for networks with “sub-sub-modules” like in resnet18 
    structure, maybe somewhere we need a recursive pass, in order to access the 
    “leaf-modules” of the graph…
    https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/13
    '''
    def __init__(self, submodule, extracted_layers):
        '''submodule: the pretrained model
        extracted_layers: the name of layers, for example, ["conv1","layer1","avgpool"]
        '''
        super(FeatureExtractor,self).__init__()
        self.submodule = submodule
        self.extracted_layers= extracted_layers
 
    def forward(self, x):
        outputs = []
        # for models trained with single gpu
        # for name, module in self.submodule._modules.items():

        # for models trained with torch.nn.DataParallel()
        # there is an extra 'module' for parallel model
        for name, module in self.submodule.module._modules.items():
            # ipdb.set_trace()

            if name is "dp": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                # print('Extracting', name)
                outputs.append(x)
            
        if len(outputs) == 0:
            print('Cannot find ' + self.extracted_layers + ' in the model.')
        return outputs


# ------------------------------------------------------------------------------
# visualization
def pred_show(x, **kwargs):
    """Show prediction results."""
    if x.ndim == 2:
        plt.imshow(x, interpolation="nearest", **kwargs)
    elif x.ndim == 1:
        plt.imshow(x[:,None].T, interpolation="nearest", **kwargs)
        plt.yticks([])
    plt.axis("tight")


def pred_viz(preds, gts, avg_acc, n_class, save_dir, save_name):
    """Visualizing the predction results of a sequence."""
    f_length, seq_length = preds.shape
    plt.figure(num=0, figsize=(30, 2*f_length))

    # normalize the labels to [0, 1]
    pred_norm = np.array(preds) / float(n_class - 1)
    gt_norm = np.array(gts) / float(n_class - 1)

    for i in range(f_length):
        # stack predicted and gt labels for comparison
        tmp = np.vstack([gt_norm[i], pred_norm[i]])
        plt.subplot(f_length, 1, i+1)
        pred_show(tmp, vmin=0, vmax=1)
        plt.yticks([])
        plt.ylabel("next #{}".format(i))
    
    # super title
    plt.suptitle('Seq2seq_len{}_{}: avg_acc {:.02f} \n (top: predictions; \
        bottom: ground truth)'.format(f_length, save_name, avg_acc))
    plt.show()
    plt.savefig(os.path.join(save_dir, save_name) )
    plt.clf()


# ------------------------------------------------------------------------------
# others
def parse_rec(root_dir, rec_file):
    date_videos = []
    rec_path = os.path.join(root_dir, rec_file)
    # annotation list: data/ikea.../videos/2016-09-01/GOPRO115.MP4.30HZ.json
    
    with open(rec_path, 'r') as f_csv:
        annot_list = csv.reader(f_csv)

        for row in annot_list:
            # full path of json
            # TO DO: move out of this function
            sub_dirs = ''.join(row).split('/')
            date_video = os.path.join(sub_dirs[-2], sub_dirs[-1][:-10])
            date_videos.append(date_video)
    return date_videos


def str2bool(v):
    '''Transfer a string to a bool parameter. Designed for bool input of parser.'''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


