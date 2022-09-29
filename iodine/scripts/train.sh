#!/bin/bash
## IMPORTANT: world_size has to be equal nproc_per_node and num_workers in train.sh
##python -m torch.distributed.launch  --nproc_per_node=4 --use_env ./train_mngr.py \
##  --mode train --parallel --device_ids 0,1,2,3 \
##  --dataset_version CLEVR6 --train_dataset_size -1 --test_dataset_size -1 --train_batch_size 16 \
##  --model_version direct_ms --kmeans_iteration 100 --slots 20 --cluster_centers 20 \
##  --img_encoder --experiment_name Encoder --world_size 4 --num_workers 4 \
##  --resume_training --resume_epoch 20 ## <-- if you want to resume training, remove the "##" and put in the resume_epoch you want to resume from


