# You can only comments at head of the bash script.
CUDA_VISIBLE_DEVICES=0 \
python train_seq2seq_oad.py \
-j 0 \
-b 1 \
--epochs 80 \
--lr 0.001 \
--lr_steps 30 \
--momentum 0.9 \
--weight_decay 5e-4 \
--root_dir '../data/OAD/' \
--model_folder 'trained_models' \
--model_exp 'seq2seq' \
--log_folder 'logs' \
--log_freq 5 \
--pw 100 \
--print_freq 2 \
--future_length 5 \
--mode 'flow' \
--pose_1080p_flag 'yes' \
--pose_flag 'yes' \
--task 'joint'

