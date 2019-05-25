nohup python main.py \
--model_type gc_resnet101 \
--batch_size 128 \
--epoches 100 \
--learning_rate 1e-3 \
--momentum 0.9 \
--optim_type SGD \
--loss_type BCE_loss \
--exp_version v8 \
--load_model_path "" & \
