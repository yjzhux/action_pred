# activity-forecasting
Forecasting human activities and analyzing temporal dynamic patterns in structured tasks, i.e., hand washing and hygiene training, assembly tasks, surgical operations and patient mobility. 

## Requirements:
- python 3
- OpenCV 3.4.x
- PyTorch 1.1.0, Tensorboard 1.14
I export the conda environment using command:

    conda env export > torch.yml

To use this conda environment: 

    cd ./dataset_public/share/      # the path of torch.yml
    conda env create -f torch.yml   # create your own conda environment 
    conda activate torch            # activate this environment


## data downloading
IkeaDB: https://tengdahan.github.io/ikea.html
OAD: http://www.icst.pku.edu.cn/struct/Projects/OAD.html 


## Structure
In this repo, we mainly use deep network to forecast the following sequence labels and poses jointly. There are four folders that are used. The rest folders and files are copied from 'iDT + LSTM' repo, but not used here.

    flownet2-pytorch/   # a pytorch version of flownet2 (https://github.com/NVIDIA/flownet2-pytorch)
    pose_gru/           # motion prediction (https://github.com/enriccorona/human-motion-prediction-pytorch)
    tools/              # data processing
    resnet18            # main workspace: training, testing, model...

All the output folders are defined as follows:

    ./data/IkeaDB/: root dir
    videos/:        original video data
    frames/:        extracted frames from videos
    flows/:         optical flow from flownet2
    data/:          CNN features from resnet18, used in training and testing
    logs/:          saving tensorboard outputs
    trained_models/:trained models (best and last)
    share/: sharing some configs


### 1 - Extract frames from videos (Done)
    tools/proc_ikea.py
    input: ikea-fa-release-data/videos
    output: ikea-fa-release-data/frames, ikea-fa-release-data/feats/poses.h5
Extract frames (poses) from original videos (annotations). Ffmpeg is used to extract frames at certain FPS (default is 30). Resample labels and poses according to the length of new frame numbers.

### 2 - Extract optical flows from frames (Done)
    flownet2-pytorch/
    sh install.sh
    test_flow.py
    input: ikea-fa-release-data/frames
    output: ikea-fa-release-data/flows (.png and .flow)
Compute optical flows between two consecutive frames.

### 3- Save data to h5 file (Done)
    tools/frame2h5.py
    input: frames or flows
    output: .h5 files in the same folder
Save each video data into one h5 file to accelerate data reading.

### 4 - Train CNN model (Done)
    src/train_fc.py    
Training classification networks (rgb and flow) which are used for feature extraction.

### 5 - Extract CNN features (Done)
    src/extract_feats.py (L57-59)
The features are saved in 'feats/feats_len5.h5'. Run twice: one for testing set and the other for training set.

### 6 - Training and Validation
    src/train_seq2seq_run.sh   # train a sequence to sequence model (train_seq2seq.py)   
    src/test_seq2seq.py        # test and save visulized poses predictions

    # creat videos from saved predictions
    cd ./dataset_public/ikea-fa-release-data/logs/YOUR-LOG-FOLDER/show/
    ffmpeg -threads 4 -y -r 30 -i 1/%4d.png seq_1.mp4