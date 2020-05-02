'''
Extract RGB frames from IKEA dtaset
'''

import os
import skvideo
import json, h5py, csv
import scipy.io as sio
import numpy as np
import ipdb

# Another method. Not used in this code.
def get_frame(video_path, max_height = 360, max_fps = 30):
    
    # Normalize video with maximum height 360
    # Set this to 30 to normalize videos at 30 fps
    outputdict = {}
    metadata = skvideo.io.ffprobe(video_path)
    width = int(metadata['video']['@width'])
    height = int(metadata['video']['@height'])
    rate = [int(x) for x in metadata['video']['@r_frame_rate'].split('/')]
    if height > max_height:
        outputdict['-s'] = str(round(width * max_height / height)) + 'x' + str(max_height)
    if rate[0] / rate[1] > max_fps:
        outputdict['-r'] = str(max_fps)

    # Load videos frame-by-frame, scale down video
    video_frames = skvideo.io.vreader(fname = video_path, 
                                      as_grey = False,
                                      outputdict=outputdict)
    ipdb.set_trace()
    return video_frames

def extract_frame_ikea(data_dir):

    video_dir = os.path.join(data_dir, 'videos')
    image_dir = os.path.join(data_dir, 'frames')
    date_list = ['2016-08-11', '2016-08-18', '2016-09-01']
    for date in date_list:
        video_date_dir = os.path.join(video_dir, date)
        image_date_dir = os.path.join(image_dir, date)
        
        for root, dirs, files in os.walk(video_date_dir):
            for name in files:
                if name.endswith('.MP4') and not name.startswith('._'):
                    image_date_video_dir = os.path.join(image_date_dir, name + '.frames')
                    if not os.path.exists(image_date_video_dir):
                        os.makedirs(image_date_video_dir)
                    video_path = os.path.join(video_date_dir, name)
                    image_path = os.path.join(image_date_video_dir, '%06d.jpg')

                    # Normalize video to 640x360
                    # Set this to 30 to normalize videos at 30 fps
                    print('video_path: ', video_path, '-->', 'image_path', image_date_video_dir)
                    os.system('ffmpeg -i ' + video_path + ' -qscale:v 2 -r 30 -s 640x360 ' + image_path)
                    # extract 1 frame out of every 2 frames: -vf 'select=not(mod(n\, 2))' -vsync 0
                    # os.system('ffmpeg -i ' + video_path + " -vf 'select=not(mod(n\, 2))' -vsync 0 -s 640x360 "+ image_path)
                    # ipdb.set_trace()
                    
def resample_annot(rows, num_new):
    '''resample annotations to match 30fps frames
    rows: original list; proportion = num_new / num_old
    '''
    counter = 0.0
    last_count = None
    results = []
    num_old = len(rows)
    proportion = float(num_new) / num_old

    # downsample
    if proportion <= 1:
        for row in rows:
            counter += proportion
            if int(counter) != last_count:
                results.append(row)
                last_count = int(counter)
        
        if len(results) > num_new:
            results = results[:num_new]
    # upsample
    else:
        diff = num_new - num_old
        segments = int(num_old / diff)
        for row in rows:
            counter += 1
            results.append(row)
            # repeat one frame annotation every #segments
            if counter >= segments:
                results.append(row)
                counter = 0
            
        if len(results) != num_new:
            print('Error: less')
            ipdb.set_trace()
            results.append(rows[num_new])
    return results

def resample_pose(poses, num_new):
    '''resample poses to match 30fps
    poses.shape = (14, 2, N)
    return shape with (num_frame, 14, 2)
    '''
    counter = 0.0
    last_count = 0
    num_joint, num_channel, num_old = poses.shape
    results = np.zeros((num_new, num_joint, num_channel), dtype=np.float32)
    proportion = float(num_new) / num_old

    # downsample
    if proportion <= 1:
        for idx in range(num_old):
            counter += proportion
            if int(counter) != last_count:
                results[last_count] = np.array(poses[:, :, idx])
                last_count = int(counter)
        
        if len(results) != num_new:
            print('Error: more{%d}'.format(num_new - last_count))
            results = np.array(results[:num_new, :, :])
    # upsample
    else:
        diff = num_new - num_old
        segments = int(num_old / diff)
        for idx in range(num_old):
            results[last_count] = np.array(poses[:, :, idx])
            counter += 1
            last_count += 1
            # repeat one frame annotation every #segments
            if counter >= segments:
                results[last_count] = np.array(poses[:, :, idx])
                counter = 0
                last_count +=1
                
            
        if len(results) != num_new:
            print('Error: less {%d}'.format(num_new - last_count))
            results[last_count] = np.array(poses[:, :, idx])
    return results

def extract_annot_ikea(data_dir):
    # Read in all annotations with fields 
    # ('video_id', 'seq_idx', 'clip_path', 'num_frames', 'frame_name', 
    # 'person_idx', 'cropbox', 'poses', 'activity_id', 'activity_labels')
    annot_path = os.path.join(data_dir, 'IkeaClipsDB.mat')
    db = sio.loadmat(annot_path, squeeze_me=True)['IkeaDB']

    # A path record starts with the prefix '/data/home/cherian/IkeaDataset/Frames/'
    clip_path_prefix = '/data/home/cherian/IkeaDataset/Frames/'
    # Go through each video
    for rec in db:
        vid_middle = rec['clip_path'][len(clip_path_prefix):]
        vid_path = os.path.join('videos', vid_middle) + '.MP4'

        # Get annotated activity labels and save to a mat file
        annotation = {}
        annotation['person_idx'] = rec['person_idx']
        Y = rec['activity_id'].tolist()
        Y_labels = [act if len(act)>0 else 'None' for act in rec['activity_labels']]
        

        # Resample frame rate to 30FPS on ground truth
        frame_dir = os.path.join(data_dir, 'frames', vid_middle) + '.MP4.frames'
        num_frames = len([name for name in os.listdir(frame_dir) if os.path.isfile(os.path.join(frame_dir, name))])
        # print('compress_rate, num_frames_new, num_frames_old: ', compress_rate, num_frames_new, num_frames_old)
        annotation['Y'] = resample_annot(Y, num_frames)
        annotation['Y_labels'] = resample_annot(Y_labels, num_frames)
        
        # ipdb.set_trace()
        # save downsampled anotations
        sio.savemat(os.path.join(data_dir + vid_path) + '.30HZ.mat', annotation)
        with open(os.path.join(data_dir + vid_path) + '.30HZ.json', 'w') as f:
            json.dump(annotation, f)

def extract_pose_ikea(data_dir):
    # Read in all annotations with fields 
    # ('video_id', 'seq_idx', 'clip_path', 'num_frames', 'frame_name', 
    # 'person_idx', 'cropbox', 'poses', 'activity_id', 'activity_labels')
    annot_path = os.path.join(data_dir, 'IkeaClipsDB.mat')
    db = sio.loadmat(annot_path, squeeze_me=True)['IkeaDB']
    # A path record starts with the prefix '/data/home/cherian/IkeaDataset/Frames/'
    clip_path_prefix = '/data/home/cherian/IkeaDataset/Frames/'
    # open the h5 file to save all pose data to one file
    save_path = os.path.join(data_dir, 'feats/poses.h5')
    

    # Go through each video
    for rec in db:
        vid_middle = rec['clip_path'][len(clip_path_prefix):]
        vid_path = os.path.join('videos', vid_middle) + '.MP4'

        # Get poses
        poses = rec['poses']
        # This is the region which we cropped out to pass to CPM. Rectangle in [x1,
        # y1, x2, y2] format.
        cb = rec['cropbox'].astype(np.float32)

        # Resample frame rate to 30FPS on ground truth
        with open(os.path.join(data_dir + vid_path) + '.30HZ.json', 'r') as f:
            annotation = json.load(f)
            labels = annotation['Y']
        num_frames = len(labels)
        new_poses = resample_pose(poses, num_frames)
        
        # Poses were produced from 368x368 warp of a crop around real location; we
        # need to move poses back into *image coordinates*.
        # ipdb.set_trace()
        w = cb[2] - cb[0]
        h = cb[3] - cb[1]
        for idx in range(num_frames):
            pose = new_poses[idx][:]
            pose = pose / 368.0 * np.array([w, h])[None, :] \
                + np.array([cb[0], cb[1]])[None, :]
            new_poses[idx] = pose[:]
        
        # save downsampled anotations / poses to h5 file
        # ipdb.set_trace()
        f = h5py.File(save_path, 'a')
        f.create_group('/'+vid_middle)
        f.create_dataset(vid_middle+'/poses', data=new_poses)
        f.create_dataset(vid_middle+'/labels', data=labels)
        f.close()
    print('Saved all pose data.')


# Ikea-Bench: ids 9 and 11 for testing
# Ikea-Ground: ids 9 and 13 for testing
def split_train_test_IkeaDB(data_dir):
    print("Split the IkeaDB train test lists")
    # IkeaDB_dir = "../data/ikea*/*/*/*.json"
    # filenames = sorted(glob.glob(IkeaDB_dir))
    jsonfiles = [os.path.join(d, f) 
                 for d, subdir, files in os.walk(data_dir) 
                 for f in files if f.endswith('.30HZ.json') and not f.startswith('._')]
    test_idx = [9, 11, 13]
    training = [f for f in jsonfiles if json.load(open(f))['person_idx'] not in test_idx]
    testing = [f for f in jsonfiles if json.load(open(f))['person_idx'] in test_idx]
    print('#total:', len(jsonfiles), ' #train:', len(training), ' #test:', len(testing))

    # Save train and test lists 
    with open(data_dir+'train_30fps.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for train in training:
            writer.writerow([train])

    with open(data_dir+'test_30fps.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for test in testing:
            writer.writerow([test])


if __name__ == '__main__':
    data_dir = '../data/ikea-fa-release-data/'

    # 1. extract frames
    # extract_frame_ikea(data_dir)
    # print('Done: video --> frames')

    # 2. resample annotations as 30fps
    # extract_annot_ikea(data_dir)
    # extract_pose_ikea(data_dir)
    
    # 3. check keys
    # save_path = os.path.join(data_dir, 'feats/poses.h5')
    # f = h5py.File(save_path, 'r')
    # for key in f.keys():
    #     print(key)
    # f.close()

    # 4. split training / testing 
    split_train_test_IkeaDB(data_dir)
    

    

    

