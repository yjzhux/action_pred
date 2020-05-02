# GPU to use
# input length of feature extractor (resnet18)
# the number of videos
# whether to use pose, must be a string from
# ('yes', 'true', 't', 'y', '1') or ('no', 'false', 'f', 'n', '0')
CUDA_VISIBLE_DEVICES=0 \
python train_lstm_multi_label.py \
-j 4 \
-b 1 \
--log_folder logs/debug \
--phase flow \
--future_length 60 \
--log_freq 5 \
--pose_flag yes
