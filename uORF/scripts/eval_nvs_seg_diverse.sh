#!/bin/bash
DATAROOT=${1:-'./uORFDatasets/room_diverse_test'}
CHECKPOINT=${2:-'./checkpoints/'}
PORT=8077
python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 500 --n_img_each_scene 4 \
    --checkpoints_dir $CHECKPOINT --name PseudoWeightsOld1 --exp_id PseudoWeightsOldGradFix --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 128 --render_size 8 --frustum_size 128 --bottom \
    --n_samp 256 --z_dim 64 --num_slots 5 --slot_init 'PseudoWeightsOld' --visualising 'True'\
    --model 'uorf_eval'
echo "Done"


