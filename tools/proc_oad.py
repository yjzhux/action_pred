"""Data pre-processing on OAD dataset.
"""
import os, sys, h5py, json, cv2, glob, ipdb , argparse
import torch
import numbers
import numpy as np

parser = argparse.ArgumentParser(description="PyTorch Spatial ")
parser.add_argument('--mode', default='rgb', type=str, metavar='S', help='rgb, flow, pose')


class Compose(object):
    """Composes several video_transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> video_transforms.Compose([
        >>>     video_transforms.CenterCrop(10),
        >>>     video_transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, video_transforms):
        self.video_transforms = video_transforms

    def __call__(self, clips):
        for t in self.video_transforms:
            clips = t(clips)
        return clips


class Scale(object):
    """ Rescales the input numpy array to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_LINEAR
    """
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clips):

        h, w, c = clips.shape
        new_w = 0
        new_h = 0
        if isinstance(self.size, int):
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return clips
            if w < h:
                new_w = self.size
                new_h = int(self.size * h / w)
            else:
                new_w = int(self.size * w / h)
                new_h = self.size
        else:
            new_w = self.size[0]
            new_h = self.size[1]

        is_color = False
        if c % 3 == 0:
            is_color = True

        if is_color:
            num_imgs = int(c / 3)
            scaled_clips = np.zeros((new_h,new_w,c))
            for frame_id in range(num_imgs):
                cur_img = clips[:,:,frame_id*3:frame_id*3+3]
                scaled_clips[:,:,frame_id*3:frame_id*3+3] = cv2.resize(cur_img, (new_w, new_h), self.interpolation)
        else:
            num_imgs = int(c / 1)
            scaled_clips = np.zeros((new_h,new_w,c))
            for frame_id in range(num_imgs):
                cur_img = clips[:,:,frame_id:frame_id+1]
                # import ipdb; ipdb.set_trace()
                # fix the bug of broadcasting input array from shape (x, y) into shape (x, y, 1)
                scaled_img = cv2.resize(cur_img, (new_w, new_h), self.interpolation)
                scaled_clips[:,:,frame_id:frame_id+1] = scaled_img[:, :, np.newaxis]
        return scaled_clips


class CenterCrop(object):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clips):
        h, w, c = clips.shape
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        is_color = False
        if c % 3 == 0:
            is_color = True

        if is_color:
            num_imgs = int(c / 3)
            scaled_clips = np.zeros((th,tw,c))
            for frame_id in range(num_imgs):
                cur_img = clips[:,:,frame_id*3:frame_id*3+3]
                crop_img = cur_img[y1:y1+th, x1:x1+tw, :]
                assert(crop_img.shape == (th, tw, 3))
                scaled_clips[:,:,frame_id*3:frame_id*3+3] = crop_img
            return scaled_clips
        else:
            num_imgs = int(c / 1)
            scaled_clips = np.zeros((th,tw,c))
            for frame_id in range(num_imgs):
                cur_img = clips[:,:,frame_id:frame_id+1]
                crop_img = cur_img[y1:y1+th, x1:x1+tw, :]
                assert(crop_img.shape == (th, tw, 1))
                scaled_clips[:,:,frame_id:frame_id+1] = crop_img
            return scaled_clips


class Parameter(object):
    '''Put all parameters related to data processing together. It is similar to
    struct in C/C++, however, python does not have a specific data structure to
    do that. So a class is used to achieve the same goal.
    '''
    def __init__(self, mode, folder, save_dir, save_name, label_dict, suffix, dt,
                C, H=224, W=224, J=15, label_folder='label', label_src='label.txt',
                label_tgt='labels.h5'):
        self.mode = mode
        self.folder = folder
        self.save_dir = save_dir
        self.save_name = save_name
        self.label_dict = label_dict
        self.suffix = suffix
        self.dt = dt
        self.C = C      # number of chennels
        self.H = H      # height
        self.W = W      # width
        self.J = J      # number of key joints in skeleton pose
        self.label_folder = label_folder
        self.label_src = label_src
        self.label_tgt = label_tgt


def frame_rename (seq_dir, folder):
    """Rename frame by frame. The lengths of file names in OAD dataset are not
    the same. For better indexing, we add '0' to the names. For instance,
    changing '18.jpg' to '000018.jpg'. Changing folder name '8' to '08'.
    """
    frame_dir = os.path.join(seq_dir, folder)
    for frame in sorted(os.listdir(frame_dir)):
        frame_path = os.path.join(frame_dir, frame)
        if os.path.isfile(frame_path):
            # change to new_name
            old_num, suffix = frame.split('.')
            new_num = old_num.zfill(6)
            new_frame = new_num + '.' +suffix
            # rename
            new_frame_path = os.path.join(frame_dir, new_frame)
            os.rename(frame_path, new_frame_path)


def frame_resize (seq_dir, source_folder, target_folder):
    """1920 x 1080 --> 640 x 360
    """
    frame_dir = os.path.join(seq_dir, source_folder)
    target_dir = os.path.join(seq_dir, target_folder)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    for frame in sorted(os.listdir(frame_dir)):
        frame_path = os.path.join(frame_dir, frame)
        target_path = os.path.join(target_dir, frame)

        if os.path.isfile(frame_path):
            # -y: overwrite
            # -nostats -loglevel 0: disable verbose
            print('source: ', source_folder, '-->', 'target', target_folder)
            os.system('ffmpeg -y -i ' + frame_path + ' -qscale:v 2 -s 640x360 ' + target_path)


def run_openpose(seq_dir, source_folder, target_folder):
    """Compute joints using openpose demo and save them to jason.
    Remember to modifiy '--num_gpu 1 --num_gpu_start 1' for your GPUs.
    """
    frame_dir = os.path.join(seq_dir, source_folder)
    img_dir = os.path.join(seq_dir, target_folder, 'img')
    json_dir = os.path.join(seq_dir, target_folder, 'json')

    # If the sequence folder already exists (the poses have been calculated), do
    # not run openpose for this sequence again.
    if os.path.isdir(img_dir):
        return  # skip what has been done
        # pass  # do nothing
    else:
        os.makedirs(img_dir)
        os.makedirs(json_dir)

    print(source_folder, '-->', target_folder)
    os.system('~/workspace/openpose_docker/run-openpose.sh' \
                + ' --image_dir ' + frame_dir \
                + ' --write_images '  + img_dir \
                + ' --write_json ' + json_dir \
                + ' --display 0 ' \
                # + ' --num_gpu 1 --num_gpu_start 1'  # using specific gpu
            )
    # ipdb.set_trace()

def read_rgb(img_path):
    """Read RGB images to numpy using opencv.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        # ipdb.set_trace()
        print("Could not load file %s" % (img_path))
        sys.exit()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def norm_flow(data, thr=20):
    '''Normalize flow data to [0, 1]. Ref:
    https://github.com/deepmind/kinetics-i3d/pull/5/files?file-filters%5B%5D=.py&file-filters%5B%5D=.sh#diff-9f76353e0bb0c0ddff674da4dfe31dc8R76
    '''
    # truncate [-thr, thr]
    data[data >= thr] = thr
    data[data <= -thr] = -thr
    # scale to [-1, 1]
    max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
    data = data / max_val(data)
    # scale to [0, 1]
    data = (data + 1) /2

    return data


def read_flow(flow_path):
    """Read flows to numpy.
    """
    FLO_TAG = 202021.25
    with open(flow_path, 'rb') as f:
        tag = np.fromfile(f, np.float32, count=1)

        if tag != FLO_TAG:
            sys.exit('Wrong tag. Invalid .flo file %s' %flow_path)
        else:
            w = int(np.fromfile(f, np.int32, count=1))
            h = int(np.fromfile(f, np.int32, count=1))
            #print 'Reading %d x %d flo file' % (w, h)

            data = np.fromfile(f, np.float32, count=2*w*h)
            # normalize to [0, 1]
            data = norm_flow(data)
            # Reshape data into 3D array (columns, rows, bands)
            flow = np.resize(data, (h, w, 2))
    return flow


def read_pose(pose_path, num_joint=15, height=360):
    '''Read joints from json outputs of openpose to numpy array.
    There are 25 joints plus 1 backgroup in BODY_25 (COCO + foot). We only need
    the first 15 joints for our model.
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#keypoint-ordering
    //     {0,  "Nose"},
    //     {1,  "Neck"},
    //     {2,  "RShoulder"},
    //     {3,  "RElbow"},
    //     {4,  "RWrist"},
    //     {5,  "LShoulder"},
    //     {6,  "LElbow"},
    //     {7,  "LWrist"},
    //     {8,  "MidHip"},
    //     {9,  "RHip"},
    //     {10, "RKnee"},
    //     {11, "RAnkle"},
    //     {12, "LHip"},
    //     {13, "LKnee"},
    //     {14, "LAnkle"},
    '''
    bad_one = []
    pose = np.zeros((num_joint, 2), dtype=np.float32)
    width = height / 9 * 16         # 360P: 360 x 640
    with open(pose_path, 'r') as f:
        data = json.load(f)         # dict, keys: version, people
        people = data['people']     # list, one item per person

        # There is only one actor in our datasets. However, openpose cannot detect
        # any person in some frames. Just keep them as all 0.
        # Know frames without any pose on them:
        #
    if len(people) == 0:
        bad_one = pose_path[61:]
        print('No person detected in: ', bad_one)
        return pose, bad_one
    else:
        actor = people[0]           # dict, keys: person_id, pose_keypoints_2d...
        # An array pose_keypoints_2d containing the body part locations and
        # detection confidence formatted as x1,y1,c1,x2,y2,c2,... The coordinates
        # x and y can be normalized to the range [0, 640] and [0, 360], while c
        # is the confidence score in the range [0,1].
        all_joints = actor['pose_keypoints_2d']  # list, len=3*25
        # extract the first 15 joints (x, y) discarding the confidence
        for kk in range(num_joint):
            pose[kk] = np.array( all_joints[3*kk : 3*kk+2] )
        # normlize (x, y) to [0, 1]
        norm_pose = pose / np.array([width, height], dtype=np.int16)
        return norm_pose, bad_one


def read_label(label_path, label_dict, start_num, seq_len):
    """Read labels as standard. A label.txt example can be:
    drinking
    120 130
    1847 1853
    eating
    207 220
    writing
    306 350
    opening cupboard
    400 435
    opening microwave oven
    730 751
    washing hands
    800 830
    sweeping
    1230 1295
    gargling
    1580 1596
    Throwing trash
    1137 1160
    1775 1786
    wiping
    1725 1747

    Note that there is a blank line at the end.
    """
    label_keys = label_dict.keys()
    labels = [0] * seq_len
    curr_action = ''

    with open(label_path, 'r') as f:
        # ipdb.set_trace()
        lines = f.readlines()

        for line in lines:
            # action name, remove '\n' at the end of the line
            line = line[:-1].lower()
            if line in label_keys:
                curr_action = line
            # durations for current action
            else:
                start, end = line.split(' ')
                duration = int(end) - int(start)
                labels[int(start)-start_num: int(end)-start_num] = \
                    ( duration * [label_dict[curr_action]] )[:]
    return labels


def data2h5(seq_dir, para, transform):
    """Save all sequences into one h5 file. The 'mode' is one of 'rgb', 'flow',
    'pose', 'label'.
    """
    # traverse all the sequences
    frame_dir = os.path.join(seq_dir, para.folder)
    frame_list = sorted( glob.glob(os.path.join(frame_dir, para.suffix)) )
    seq_len = len(frame_list) - 1 # remove the last frame which is broken
    # start_num = int( frame_list[0].split('.')[0][-6:] )
    # the format of json is different from those of jpg and flow, changed as
    # follows. Note that there is an extra '*' at the start of suffix.
    # ipdb.set_trace()
    lens = len(para.suffix) - 1
    start_num = int( frame_list[0][-lens-6 : -lens] )

    # read labels
    label_path = os.path.join(seq_dir, para.label_folder, para.label_src)
    labels = read_label(label_path, para.label_dict, start_num, seq_len)

    # create an empty array to collect every frame data
    # ipdb.set_trace()
    without_pose = []
    if para.mode in ['pose', 'pose_1080p']:
        frame_seq = np.zeros((seq_len, para.J, para.C), dtype=para.dt)
    else:
        frame_seq = np.zeros((seq_len, para.H, para.W, para.C), dtype=para.dt)

    # read data frame by frame
    for idx in range(seq_len):
        frame_path = frame_list[idx]
        if para.mode == 'rgb':
            frame = read_rgb(frame_path)
        elif para.mode == 'flow':
            frame = read_flow(frame_path)
        elif para.mode == 'pose':
            frame, bad_one = read_pose(frame_path, num_joint=15, height=360)
            # record bad frames that have no pose detected
            if len(bad_one) > 0:
                without_pose.append(bad_one)
        elif para.mode == 'pose_1080p':
            frame, bad_one = read_pose(frame_path, num_joint=15, height=1080)
            # record bad frames that have no pose detected
            if len(bad_one) > 0:
                without_pose.append(bad_one)
        else:
            print("Only support 'rgb', 'flow', 'pose' and 'pose_1080p'.")

        # apply transforms to the img/flow data, but not fot pose
        if para.mode in ['pose', 'pose_1080p']:
            pass
        else:
            frame = transform(frame)

        # put data of all frames into a sequence
        frame_seq[idx] = frame[:]

    # save data to h5 file
    compression = 32001
    compression_opts = (0, 0, 0, 0, 9, False, 1)

    seq = seq_dir.split('/')[-1]
    h5path = os.path.join(para.save_dir, para.save_name)
    f = h5py.File(h5path, 'a')
    grp = save_meta(f, seq, labels, start_num, seq_len)
    grp.create_dataset(mode, data=frame_seq, compression=compression, \
                                 compression_opts=compression_opts)
    # You're reading in Unicode strings, but specifying your datatype as ASCII.
    # According to the h5py wiki, h5py does not currently support this conversion.
    # You'll need to encode the strings in a format h5py handles:
    ascii_list = [n.encode("ascii", "ignore") for n in without_pose]
    grp.create_dataset('without_pose', data=ascii_list)
    f.close()

    # save additional labels.h5 for fast reference
    label_h5path = os.path.join(para.save_dir, para.label_tgt)
    label_f = h5py.File(label_h5path, 'a')
    label_grp = save_meta(label_f, seq, labels, start_num, seq_len)
    label_f.close()


def save_meta(f, seq, labels, start_num, seq_len):
    '''meta data: label, start_num, seq_len
    '''
    grp = f.create_group(seq)
    grp.create_dataset('label', data=labels)
    grp.create_dataset('start_num', data=start_num)
    grp.create_dataset('seq_len', data=seq_len)

    return grp


def rm_h5(file_dir, file_name):
    '''Delete h5 file if it already exists.
    '''
    file_path = os.path.join(file_dir, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        print('Delete', file_name)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    mode = args.mode
    root_dir = '../data/OAD'
    data_dir = os.path.join(root_dir, 'data')
    save_dir = os.path.join(root_dir, 'group_data')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        print(save_dir, 'created.')

    # parameters for data processing
    if mode == 'rgb':
        save_name = 'rgbs.h5'
        folder = 'frames'
        suffix = '*.jpg'
        dt = np.uint8
        C = 3
    elif mode == 'flow':
        save_name = 'flows.h5'
        folder = 'flows'
        suffix = '*.flow'
        dt = np.float32
        C = 2
    elif mode == 'pose':
        save_name = 'poses.h5'
        folder = 'poses/json'
        suffix = '*_keypoints.json'
        dt = np.float32
        C = 2
    elif mode == 'pose_1080p':
        save_name = 'poses_1080p.h5'
        folder = 'poses_1080p/json'
        suffix = '*_keypoints.json'
        dt = np.float32
        C = 2
    else:
        print('Invalid mode! Only support rgb, flow and pose.')
        sys.exit()
    # The relation dict between names and labels of annotations
    label_dict = {
        'none':                     0,
        'drinking':                 1,
        'eating':                   2,
        'writing':                  3,
        'opening cupboard':         4,
        'washing hands':            5,
        'opening microwave oven':   6,
        'sweeping':                 7,
        'gargling':                 8,
        'throwing trash':           9,
        'wiping':                   10
    }

    para = Parameter( mode, folder, save_dir, save_name, label_dict, suffix, dt, C)
    transform = Compose([Scale(para.H), CenterCrop(para.H)])
    # print(para.__dict__)
    print('Current working mode: ', mode)

    rm_h5(save_dir, para.save_name) # for rgbs.h5, flows.h5, poses.h5
    rm_h5(save_dir, para.label_tgt) # for labels.h5

    # traverse all sequence folders
    for seq in sorted(os.listdir(data_dir)):

        seq_dir = os.path.join(data_dir, seq)
        print('Processing seq: ' + seq)
        # ipdb.set_trace()

        # ----------------------------------------------------------------------
        # 1. rename files for better indexing
        # frame_rename(seq_dir, 'color')
        # frame_rename(seq_dir, 'depth')
        # frame_rename(seq_dir, 'skeleton')

        # ----------------------------------------------------------------------
        # 2. resize 1080p images to 360p
        # frame_resize(seq_dir, 'color', 'frames')

        # ----------------------------------------------------------------------
        # 3. Compute joints using openpose demo and save them to json.
        #    Remember to modifiy '--num_gpu 1 --num_gpu_start 1' for your GPUs.
        #    Install Docker first. https://docs.docker.com/engine/install/ubuntu/
        #    Docker modified from Andreas
        # run_openpose(seq_dir, 'color', 'poses_1080p')
        # run_openpose(seq_dir, 'frames', 'poses')

        # ----------------------------------------------------------------------
        # 4. save all sequences to one h5 file
        #    The 'mode' is one of 'rgb', 'flow', 'label', 'pose', 'pose_1080p'.
        data2h5(seq_dir, para, transform)




