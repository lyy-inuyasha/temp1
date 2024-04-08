cp `basename $0` a.sh

lr=0.00001
epochs=200
bs=8

CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port 47711 train_soft.py \
 --epochs ${epochs}\
 --bs ${bs} \
 --lr ${lr} \
 --log_dir "logs/sfcn-gm_lr${lr}_bs${bs}" \
 --tag 'sfcn baseline, gray matter, 0.5 dropout, without image resize'
