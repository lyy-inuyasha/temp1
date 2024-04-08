CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 47708 train.py \
 --epochs 100 \
 --bs 16 \
 --lr 0.001 \
 --log_dir 'logs/noisy-baseline_lr0.001_bs16'
