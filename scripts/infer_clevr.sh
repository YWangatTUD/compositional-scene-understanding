CUDA_VISIBLE_DEVICES=0 python infer_clevr.py \
--learning_rate 4e-3 \
--scheduler_config configs/clevr-2D-pos/scheduler/scheduler_config.json \
--unet_config configs/clevr-2D-pos/unet/config.json \
--dataset_root /space/ywang86/CLEVR_v1.0 --resolution 64
