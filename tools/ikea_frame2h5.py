import os, sys, h5py, json, cv2, glob, ipdb 
import torch
import numbers
import numpy as np 


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


def read_rgb(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        # ipdb.set_trace()
        print("Could not load file %s" % (img_path))
        sys.exit()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_flow(filename):
    FLO_TAG = 202021.25
    with open(filename, 'rb') as f:
        tag = np.fromfile(f, np.float32, count=1)
        
        if tag != FLO_TAG:
            sys.exit('Wrong tag. Invalid .flo file %s' %filename)
        else:
            w = int(np.fromfile(f, np.int32, count=1))
            h = int(np.fromfile(f, np.int32, count=1))
            #print 'Reading %d x %d flo file' % (w, h)
                
            data = np.fromfile(f, np.float32, count=2*w*h)

            # Reshape data into 3D array (columns, rows, bands)
            flow = np.resize(data, (h, w, 2))

    return flow

def read_label(annot_path):
    with open(annot_path, 'r') as f_json:
        annot = json.load(f_json)
        labels = annot['Y']
    return labels


def frame2h5(video_date_dir, frame_date_dir, transform):
    H, W, C = 224, 224, 3
    for root, dirs, files in os.walk(video_date_dir):
        for file in files:
            if file.endswith('.30HZ.json') and not file.startswith('._'):
                folder = file[:-10]+'.frames'
                # ipdb.set_trace()

                # read labels
                print(file)
                label_path = os.path.join(video_date_dir, file)
                labels = read_label(label_path)

                # read videos frame by frame
                file_path = os.path.join(frame_date_dir, folder, "*.jpg")
                file_list = glob.glob(file_path)
                L = len(file_list)
                frame_seq = np.zeros((L, H, W, C), dtype=np.uint8)
                
                for t in range(L):
                    frame_path = os.path.join(frame_date_dir, folder, "%06d.jpg" %(t+1))
                    frame = read_rgb(frame_path)
                    frame = transform(frame)
                    frame_seq[t] = frame[:]
                # frame_seq = transform(frame_seq)

                
                # save to h5 file
                h5path = os.path.join(frame_date_dir, folder + '.h5')
                f = h5py.File(h5path, 'w')
                f.create_dataset('data', data=frame_seq, compression='gzip')
                f.create_dataset('label', data=labels)
                f.close()

def flow2h5(video_date_dir, flow_date_dir, transform):
    H, W, C = 224, 224, 2
    for root, dirs, files in os.walk(video_date_dir):
        for file in files:
            if file.endswith('.30HZ.json') and not file.startswith('._'):
                folder = file[:-10]+'.flows'
                
                # read labels
                print(file)
                label_path = os.path.join(video_date_dir, file)
                labels = read_label(label_path)

                # read videos frame by frame
                file_path = os.path.join(flow_date_dir, folder, "*.flow")
                file_list = glob.glob(file_path)
                L = len(file_list)
                flow_seq = np.zeros((L, H, W, C), dtype=np.float16)
                
                for t in range(L):
                    # ipdb.set_trace()
                    flow_path = os.path.join(flow_date_dir, folder, "%06d.flow" %(t+1))
                    flow = read_flow(flow_path)
                    flow = transform(flow)
                    flow_seq[t] = flow[:]
                    
                
                # save to h5 file
                h5path = os.path.join(flow_date_dir, folder + '.h5')
                f = h5py.File(h5path, 'w')
                f.create_dataset('data', data=flow_seq, compression='gzip')
                f.create_dataset('label', data=labels)
                f.close()

if __name__ == '__main__':
    data_dir = '../data/ikea-fa-release-data/'
    dates = ['2016-08-11', '2016-08-18', '2016-09-01']
    
    transform = Compose([Scale(224), CenterCrop(224)])
    for date in dates:
        video_date_dir = os.path.join(data_dir, 'videos', date)
        frame_date_dir = os.path.join(data_dir, 'frames', date)
        flow_date_dir = os.path.join(data_dir, 'flows', date)

        # 1. save frames to h5
        frame2h5(video_date_dir, frame_date_dir, transform)
    
        # 2. save flow to h5
        flow2h5(video_date_dir, flow_date_dir, transform)

          
