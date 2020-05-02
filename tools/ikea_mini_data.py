# The number of frames is about 340k at 30fps:  300k for training while 40k for testing.
# It takes about 30 mins to train on the full dataset. So I create a mini dataset which 
# consists of 8 training videos and 2 testing videos.

import csv
import os
from random import sample

def select_video(full_data, mini_data, num_videos):
    records = []
    with open(full_data, 'r') as f_r:
        reader = csv.reader(f_r)
        for row in reader:
            records.append(row)
        # import ipdb; ipdb.set_trace()
        idx_list = sample(range(len(records)), num_videos)
        
        with open(mini_data, 'w') as f_w:
            writer = csv.writer(f_w)
            for idx in idx_list:
                writer.writerow(records[idx])

if __name__ == "__main__":
    full_data = '../data/ikea-fa-release-data/test_30fps.csv'
    mini_data = '../data/ikea-fa-release-data/test_30fps_mini.csv'
    num_videos = 2
    select_video(full_data, mini_data, num_videos)
