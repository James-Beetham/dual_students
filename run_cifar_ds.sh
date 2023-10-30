cd dfme;

python3 train.py \
    --dataset cifar10 \
    --data_dir ~/data \
    --ckpt checkpoint/teacher/cifar10-resnet34_8x.pt \
    --cuda 0 \
    --grad_m 1 \
    --query_budget 20 \
    --log_dir save_results/cifar10 \
    --num_students 2 \
    --lr_G 1e-4 \
    --lr_S 0.3 \
    --student_model resnet18_8x \
    --loss l1 \
    --epoch_itrs -1 \
    --resume \
