#!/bin/bash
source scripts/all_models.sh
DATA_TYPES=("aime" "math500")
NUM_SAMPLES=16

# Generate commands for each model and data type combination
job_id=0
for model in "${MODELS[@]}"; do
    for data_type in "${DATA_TYPES[@]}"; do
        # for type_flag in "original" "modified"; do
        for type_flag in "modified"; do
            CUDA_VISIBLE_DEVICES=0,1 python3 infer.py --model "$model" --num_samples "$NUM_SAMPLES" --tensor_parallel_size 2 --data_type "$data_type" --type_flag "$type_flag" --cot
        done
    done
done 