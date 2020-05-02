# You can only comments at head of the bash script.
# phase: rgb, flow
CUDA_VISIBLE_DEVICES=2 \
python train_fc.py \
-j 4 \
-b 128 \
--epochs 80 \
--lr 0.001 \
--lr_steps 30 \
--momentum 0.9 \
--weight_decay 5e-4 \
--root_dir '../data/ikea-fa-release-data/' \
--model_folder 'trained_models' \
--log_folder 'logs' \
--model_exp 'fc' \
--phase 'rgb' \
--frame_length 5 \
--dropout 0.7

