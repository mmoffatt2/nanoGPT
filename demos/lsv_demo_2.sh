#!/bin/bash


# Learned Dataset Embedding
## very small footprint, only an addition
## applies best to layer 0
bash data/shakepeare_char/get_dataset.sh

pushd data/opus-100
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  python3 get_dataset.py
  prepare.py -t input.txt --method tiktoken
else
  echo "train.bin, val,bin, and meta.pkl already found for opus-100"  
fi
popd

python3 train.py \
  --training_mode multidataset \
  --multidataset_wte \
  --dataset shakespeare_char \
  --dataset_list  shakespeare_char opus-100 \
  --dataset_sampling_probs 1 10  \
  --dataset_sampling_probs_final 1 1 \
  --dataset_sampling_probs_transition_method cosine \
  --dataset_interleaving \
  --batch_size 8 \
  --learning_rate "6e-4" \
  --decay_lr \
  --min_lr "6e-5" \
  --dropout "0.1" \
  --n_layer 12 \
  --n_head 12 \
  --n_embd 768 \
  --block_size 256 \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --max_sample_tokens 100 \
  --max_iters 10000 \
  --warmup_iters 1000
  --eval_interval 1000 \
  --sample_each_eval \
  --init_from "scratch" \
  --use_lsv \
  --apply_lsv_at_layer_idx 0 \
  --lsv_variant "one_hot" \
  --out_dir "out_scratch_multidataset_one_hot" \
  --tensorboard_run_name "out_one_hot" \
  --compile
