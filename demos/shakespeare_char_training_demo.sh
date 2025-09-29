#!/bin/bash
# shakespeare_char_training_demo.sh
# Demonstrates end-to-end training on the shakespeare_char dataset with a lightweight configuration.

set -e

# Ensure the dataset is prepared.
pushd data/shakespeare_char > /dev/null
bash get_dataset.sh
popd > /dev/null

OUT_DIR="out/shakespeare_char_training_demo"
mkdir -p "${OUT_DIR}"

# Train a compact GPT on the dataset.
python3 train.py \
  --dataset shakespeare_char \
  --out_dir "${OUT_DIR}" \
  --max_iters 500 \
  --eval_interval 50 \
  --block_size 128 \
  --batch_size 24 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --learning_rate 3e-4 \
  --dropout 0.1 \
  --compile \
  --no-tensorboard_log

# Show where the training artifacts live.
echo "Training complete. Checkpoints and logs are in ${OUT_DIR}."
