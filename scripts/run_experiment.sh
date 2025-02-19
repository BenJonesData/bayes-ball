#!/bin/bash

poetry run python src/experiment.py \
    --training_data_path 'data/train_data.csv' \
    --validation_data_path 'data/val_data.csv'\
    --learning_rate 0.001 \
    --num_epochs 10000 \
    --num_samples 1000 \
    --num_batches 1 \
    --league_tag 'epl'
    # --hidden_units  