#!/bin/bash

declare -A models
# models[48]='esm2_t48_15B_UR50D'
# models[36]='esm2_t36_3B_UR50D'
models[33]='esm2_t33_650M_UR50D'
models[30]='esm2_t30_150M_UR50D'
models[12]='esm2_t12_35M_UR50D'
models[6]='esm2_t6_8M_UR50D'

# Function to generate a space-separated sequence from 0 up to n
get_layers() {
    seq 0 $(($1)) | tr '\n' ' '
}

for dataset in train validation test; do
    for model in "${!models[@]}"; do
        layers=$(get_layers $model)
        esm-extract "${models[$model]}" "/shared/stability/$dataset.fasta" "/shared/stability/${models[$model]}/$dataset" --repr_layers $layers --include mean
    done
done