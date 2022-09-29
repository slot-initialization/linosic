#!/bin/bash
#python3 -m eval_mngr --mode eval --device_ids 0 \
#  --dataset_version CLEVR6 --test_dataset_size 500 \
#  --model_version pseudoweights --kmeans_iteration 10 --slots 7 --cluster_centers 14 \
#  --world_size 1 --num_workers 1 --experiment_name Encoder --img_encoder \
#  --eval_checkpoint_number 980 --eval_best_checkpoint\
#  --save_generated_images 500 \

#python3 -m eval_mngr --mode eval --device_ids 0 \
#    --dataset_version CLEVR10 --test_dataset_size 500 \
#    --model_version mlp_ms --kmeans_iteration 100 --slots 20 --cluster_centers 20 \
#    --img_encoder --experiment_name Encoder --world_size 1 --num_workers 1 \
#    --eval_checkpoint_number 260 --save_generated_images 500 --eval_best_checkpoint --sort_slots \

##  --eval_best_checkpoint \ <-- to evaluate best checkpoint
##--eval_checkpoint_number 60 \ <-- to evaluate specific checkpoint
##  --img_encoder --experiment_name Encoder <-- if your model uses an image encoder
##  For the other arguments only change dataset_version, model_version, experiment_name, they have to be exactly the same
##  as the training version
