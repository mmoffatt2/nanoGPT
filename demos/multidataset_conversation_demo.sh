#!/bin/bash


# running the training
# python3 optimization_and_search/run_experiments.py --config explorations/multidataset.json --output_dir out_multi_zh
python3 train.py \
    --dataset databricks-dolly-15k \
    --training_mode multidataset \
    --log_interval 1 \
    --batch_size 64 \
    --dataset_list minipile databricks-dolly-15k \
    --dataset_sampling_probs 50 1 \
    --dataset_sampling_probs_final 1 10 \
    --dataset_sampling_probs_transition_method "linear" \
    --max_iters 20000 \
    --eval_interval 1000 \
    --sample_start_tokens $'\n\n#U:\nWhat is a good vacation plan?\n#B:\n' \
    --max_sample_tokens 256 \
    --loss_fn focal \
    --focal_gamma 2.0 \
    --init_from "scratch"
    # --gns_type exact
