#!/bin/bash

# Datasets
datasets=("/shared/stability" "/shared/fluorescence")

# Number of layers to test
layers=(6 12 30 33)

# Random seeds
seeds=(13141 51517)

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    # Loop through each layer size
    for layer in "${layers[@]}"; do
        # Loop through each seed
        for seed in "${seeds[@]}"; do
            # Execute the full_train.py script with the given parameters
            python full_train.py $layer $dataset --seed $seed &
        done
    done
done

wait


