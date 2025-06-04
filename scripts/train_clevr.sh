CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes=2 \
--main_process_port 28100 train_clevr.py --enable_xformers_memory_efficient_attention \
--dataloader_num_workers 4 --learning_rate 2e-5 --mixed_precision fp16 --num_validation_images 10 \
--val_batch_size 10 --max_train_steps 500000 --checkpointing_steps 10000 --checkpoints_total_limit 1 \
--gradient_accumulation_steps 1 --seed 42 \
--output_dir ./outputs/ \
--scheduler_config configs/clevr-2D-pos/scheduler/scheduler_config.json \
--unet_config configs/clevr-2D-pos/unet/config.json \
--dataset_root /space/ywang86/CLEVR_v1.0 --train_batch_size 128 \
--resolution 64 --validation_steps 1000 --tracker_project_name clevr

