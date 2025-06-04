CUDA_VISIBLE_DEVICES=0 python infer_celeba.py \
--scheduler_config configs/clevr-2D-pos/scheduler/scheduler_config.json \
--unet_config configs/clevr-2D-pos/unet/config.json \
--dataset_root /space/ywang86/celeba_labels