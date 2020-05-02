# joint classification and regression

CUDA_VISIBLE_DEVICES=1 \
python train_seq2seq.py \
-j 4 \
-b 1 \
--epochs 80 \
--lr 0.001 \
--lr_steps 30 \
--momentum 0.9 \
--weight_decay 5e-4 \
--root_dir '../data/ikea-fa-release-data/' \
--model_folder 'trained_models' \
--model_exp 'seq2seq' \
--log_folder 'logs' \
--pw 100 \
--phase 'flow' \
--future_length 5 \
--pose_flag 'yes' \
--task 'joint'