# You can only comments at head of the bash script.
CUDA_VISIBLE_DEVICES=0 \
python train_fc_oad.py \
-j 4 \
-b 128 \
--epochs 80 \
--model_exp 'fc' \
--lr 0.001 \
--lr_steps 30 \
--momentum 0.9 \
--weight_decay 5e-4 \
--root_dir '../data/OAD/' \
--model_folder 'trained_models' \
--model_exp 'fc' \
--log_folder 'logs' \
--mode 'flow' \
--frame_length 5 \
--pose_flag no

