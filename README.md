# Compositional Scene Understanding through Inverse Generative Modeling

We propose a compositional inverse generative modeling (IGM) framework for visual concept inference with zero-shot generalization capabilities. Our approach infers visual concepts by optimizing the conditional parameters of a compositional generative model, enabling interpreting images that are more complex than those seen during training.

![](sample_images/teaser_inference.gif)

### [Project Page](https://energy-based-model.github.io/compositional-inference/) | [Paper](https://arxiv.org/abs/2505.21780)


<hr>

This is the official codebase for **Compositional Scene Understanding through Inverse Generative Modeling**.

[Compositional Scene Understanding through Inverse Generative Modeling]()
    <br>
    [Yanbo Wang](https://ywangattud.github.io/website/) <sup>1</sup>,
    [Justin Dauwels](https://scholar.google.com/citations?user=dboVuDYAAAAJ&hl=en) <sup>1</sup>,
    [Yilun Du](https://yilundu.github.io) <sup>2</sup>
    <br>
    <sup>1</sup>TU Delft, <sup>2</sup>Harvard
    <br>

<hr>

## Setup

Run the following to create and activate a conda environment:
```
conda env create -f environment.yml
conda activate IGM
```
--------------------------------------------------------------------------------------------------------

## Training
To train a compositional generative model on CLEVR, run the following:
```
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
```
To train a compositional generative model on CelebA, run the following:
```
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes=2 \
--main_process_port 28100 train_celeba.py --enable_xformers_memory_efficient_attention \
--dataloader_num_workers 4 --learning_rate 2e-5 --mixed_precision fp16 --num_validation_images 10 \
--val_batch_size 10 --max_train_steps 50000 --checkpointing_steps 10000 --checkpoints_total_limit 10 \
--gradient_accumulation_steps 1 --seed 42 \
--output_dir ./outputs \
--scheduler_config configs/celeba/scheduler/scheduler_config.json \
--unet_config configs/celeba/unet/config.json \
--dataset_root /space/ywang86/celeba_labels --train_batch_size 64 \
--resolution 128 --validation_steps 1000 --tracker_project_name celeba
```
--------------------------------------------------------------------------------------------------------

## Inference

To infer object locations, run the following:
```
CUDA_VISIBLE_DEVICES=0 python eval_clevr.py \
--learning_rate 4e-3 \
--scheduler_config configs/clevr-2D-pos/scheduler/scheduler_config.json \
--unet_config configs/clevr-2D-pos/unet/config.json \
--dataset_root /space/ywang86/CLEVR_v1.0 --resolution 64
```

To infer facial attributes, run the following:
```
CUDA_VISIBLE_DEVICES=0 python eval_celeba.py \
--scheduler_config configs/clevr-2D-pos/scheduler/scheduler_config.json \
--unet_config configs/clevr-2D-pos/unet/config.json \
--dataset_root /space/ywang86/celeba_labels  \
--resolution 128
```

To infer animal categories with pretrained SD, run the following:
```
CUDA_VISIBLE_DEVICES=0 python eval_animal.py
```

<hr>

## Data

See our paper for details on training datasets.

|     Dataset     |                                   Link                                    | 
|:---------------:|:-------------------------------------------------------------------------:| 
|      CLEVR      |        [Link](https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip)        
|     CelebA      | [Link](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data) 
|     Animal      |                        Provided in the data folder                        |
