#!/bin/bash

# Import models from all_models
source scripts/all_models.sh

for TYPE_FLAG in "modified"; do
    for DATA in "math500" "aime"; do
        for MODEL in "${MODELS[@]}"; do
            python3 eval_pipeline.py --data "$DATA" --model "$MODEL" --type_flag "$TYPE_FLAG"
        done
    done
done