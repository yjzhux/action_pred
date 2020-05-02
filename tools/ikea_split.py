'''
This script saves annotatios and features for each individual video
in a dictionary with the following format:
- filename: 'video_name.mat' 
- X: an array of feature vectors extracted per frame
- Y: a list of activity id per frame 
- Y_labels: activity labels  
'''
import os
import os.path
import numpy as np
import scipy.io as sio
import json
import glob
import csv
import pickle
import utils
import gzip
import random

###################### Load IkeaDB #################
# CCT dataset: 2, 6, 9, 13, 17, 25 for testing
def split_train_test_CCT(vid_dir):
    print("Split the train test lists")

# Ikea-Bench: ids 9 and 11 for testing
# Ikea-Ground: ids 9 and 13 for testing
def split_train_test_IkeaDB(data_dir):
    print("Split the IkeaDB train test lists")
    # IkeaDB_dir = "dataset_public/ikea*/*/*/*.json"
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

def extract_annot_IkeaDB(data_dir):
    # Read in all annotations with fields 
    # ('video_id', 'seq_idx', 'clip_path', 'num_frames', 'frame_name', 
    # 'person_idx', 'cropbox', 'poses', 'activity_id', 'activity_labels')
    db = sio.loadmat('dataset_public/ikea-fa-release-data/IkeaClipsDB.mat', squeeze_me=True)['IkeaDB']

    # Go through each video
    clip_path_prefix = '/data/home/cherian/IkeaDataset/Frames/'
    for rec in db:
        vid_middle = rec['clip_path'][len(clip_path_prefix):]
        vid_path = os.path.join('videos', vid_middle) + '.MP4' 

        # Get annotated activity labels and save to a mat file
        annotation = {}
        annotation['person_idx'] = rec['person_idx']
        annotation['Y'] = rec['activity_id'].tolist()
        annotation['Y_labels'] = [act if len(act)>0 else 'None' for act in rec['activity_labels']]
        # sio.savemat(os.path.join(data_dir + vid_path) + '.mat', annotation)
        with open(os.path.join(data_dir + vid_path) + '.json', 'w') as f:
            json.dump(annotation, f)

##################### Get X and Y ###################
def get_XY(X_list, Y_list):
    frames = []
    fvs = []
    for f in X_list:
        if f['values'].size != 0:
            frames.append(f['frame_num'])
            fv = f['values'].astype(np.float32)
            if len(fv.shape) > 1:
                fv = fv.reshape(fv.shape[0]*fv.shape[1])
            fvs.append(fv)
            
    X = np.array(fvs)
    # Resample frame rate to 30FPS on ground truth
    compress_rate = round(len(Y_list)/len(X))
    print('compress_rate, len(X), len(Y_list): ', compress_rate, len(X), len(Y_list))
    Y = np.array([Y_list[f*compress_rate] for f in frames if f*compress_rate < len(Y_list)])
    X = X[:len(Y)]
    print('compress_rate, len(X), len(Y): ', compress_rate, len(X), len(Y))
            
    return X, Y

def get_files(vid_list, feature_type):
    with open(vid_list) as f:
        f_list = f.readlines()
    # files = [f.rstrip()[:-4]+feature_type+'.pkl' for f in f_list]
    files = [f.rstrip()[:-4]+feature_type+'.pkl.gz' for f in f_list]
    
    return files

def load_split(vid_list, feature_type = '30_minDS5_cameraTrue_scale1.41_diag_GMM_IDTFs_256_0', num_vid = 15, sample_rate = 1):  
    # Get all features
    files_features = get_files(vid_list, feature_type)

    print("Before: ", len(files_features))
    if num_vid < len(files_features):
        files_ix = random.sample(range(len(files_features)), num_vid)
        files_features = [files_features[i] for i in files_ix]
    print("After: ", len(files_features))

    X_all, Y_all = [], []
    for f_feature in files_features:
        print(f_feature)
        # Resampled 30FPS features stored as 'pkl.gz'
        with gzip.open(f_feature, 'rb') as f:
            data_tmp = pickle.load(f)
        # with open(f_feature, 'rb') as f:
        #     data_tmp = pickle.load(f)
        X, Y = get_XY(data_tmp[feature_type], data_tmp["Y"])
        print(X.shape, Y.shape)
        X_all += [X]
        Y_all += [Y]

    # Subsample the data
    if sample_rate > 1:
        X_all, Y_all = utils.subsample(X_all, Y_all, sample_rate, dim=0)

    return X_all, Y_all


def main():
    IkeaDB_dir = "dataset_public/ikea-fa-release-data/"
    # extract_annot_IkeaDB(IkeaDB_dir)
    split_train_test_IkeaDB(IkeaDB_dir)

if __name__ == '__main__':
    main()

