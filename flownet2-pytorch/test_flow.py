# python lib
import os, argparse, glob, cv2, math
from datetime import datetime
import numpy as np
import ipdb

# torch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# custom lib
import networks
from models import FlowNet2
import utils


parser = argparse.ArgumentParser(description='optical flow estimation')

# testing options
parser.add_argument('-model',           type=str,     default="FlowNet2",   help='Flow model name')
parser.add_argument('-data_dir',        type=str,     default='../data/ikea-fa-release-data',       help='path to data folder')
parser.add_argument('-list_dir',        type=str,     default='lists',      help='path to list folder')
parser.add_argument('-gpu',             type=int,     default=0,            help='gpu device id')
parser.add_argument('-cpu',             action='store_true',                help='use cpu?')

opts = parser.parse_args()

# update options
# opts.cuda = 0
opts.cuda = (opts.cpu != True)
opts.grads = {} # dict to collect activation gradients (for training debug purpose)

# FlowNet options
opts.rgb_max = 1.0
opts.fp16 = False

print(opts)

if opts.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without -cuda")

### initialize FlowNet
print('===> Initializing model from %s...' %opts.model)
model = FlowNet2(opts)

### load pre-trained FlowNet
model_filename = os.path.join(opts.data_dir, "trained_models", "%s_checkpoint.pth.tar" %opts.model)
print("===> Load %s" %model_filename)
checkpoint = torch.load(model_filename)
# ipdb.set_trace()
model.load_state_dict(checkpoint['state_dict'])

device = torch.device("cuda" if opts.cuda else "cpu")
model = model.to(device)
model.eval()

def resize_img(img, size_multiplier = 64):
    # resize image
    H_orig = img.shape[0]
    W_orig = img.shape[1]

    H_sc = int(math.ceil(float(H_orig) / size_multiplier) * size_multiplier)
    W_sc = int(math.ceil(float(W_orig) / size_multiplier) * size_multiplier)
    
    img = cv2.resize(img, (W_sc, H_sc))
    return img, W_orig, H_orig

def compute_flow(model, frame_dir, flow_dir):
    # make folders
    if not os.path.isdir(flow_dir):
        os.makedirs(flow_dir)
    
    # compute optical flow for all frames in a video
    # ipdb.set_trace()
    frame_list = sorted( glob.glob(os.path.join(frame_dir, "*.jpg")) )
    start_num = int( frame_list[0].split('.')[-2][-6:] )
    
    for t in range(start_num, start_num + len(frame_list) - 1):
        # load input images 
        img1 = utils.read_img(os.path.join(frame_dir, "%06d.jpg" %(t)))
        img2 = utils.read_img(os.path.join(frame_dir, "%06d.jpg" %(t + 1)))
        
        # resize images
        img1, W_orig, H_orig = resize_img(img1)
        img2, _, _ = resize_img(img2)
    
        with torch.no_grad():

            # convert to tensor
            img1 = utils.img2tensor(img1).to(device)
            img2 = utils.img2tensor(img2).to(device)
    
            # compute fw flow
            fw_flow = model(img1, img2)
            fw_flow = utils.tensor2img(fw_flow)

        # resize flow
        fw_flow = utils.resize_flow(fw_flow, W_out = W_orig, H_out = H_orig)

        # save rgb flow
        output_filename = os.path.join(flow_dir, "%06d.jpg" % t)
        flow_rgb = utils.flow_to_rgb(fw_flow)
        utils.save_img(flow_rgb, output_filename)
        output_flowname = os.path.join(flow_dir, '%06d.flow' % t)
        utils.save_flo(fw_flow, output_flowname)
        # ipdb.set_trace()

if __name__ == "__main__":

    # ----------IKEA dataset----------
    # frame_dir = os.path.join(opts.data_dir, 'frames')
    # flow_dir = os.path.join(opts.data_dir, 'flows')
    # date_list = ['2016-08-11', '2016-08-18', '2016-09-01']

    # for date in date_list:
    #     frame_date_dir = os.path.join(frame_dir, date)
    #     flow_date_dir = os.path.join(flow_dir, date)

    #     vframe_list = os.listdir(frame_date_dir)
    #     for vframe_name in vframe_list:
    #         if vframe_name.endswith('.frames'):
    #             vframe_dir = os.path.join(frame_date_dir, vframe_name)
    #             vflow_dir = os.path.join(flow_date_dir, vframe_name[:-6] + 'flows')
    #             # ipdb.set_trace()
    #             compute_flow(model, vframe_dir, vflow_dir)
    
    # ----------OAD dataset----------
    data_dir = '../data/OAD/data/'
    frame_folder = 'frames'
    flow_folder = 'flows'
    # traver all sequence folders
    for seq in os.listdir(data_dir):
        seq_dir = os.path.join(data_dir, seq)
        if os.path.isdir(seq_dir):
            print('Processing seq: ' + seq)
            frame_dir = os.path.join(seq_dir, frame_folder)
            flow_dir = os.path.join(seq_dir, flow_folder)

            # compute optical flow
            # ipdb.set_trace()
            compute_flow(model, frame_dir, flow_dir)

