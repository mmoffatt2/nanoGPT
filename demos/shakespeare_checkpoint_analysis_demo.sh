#!/bin/bash
# shakespeare_checkpoint_analysis_demo.sh
# Demonstrates training on the shakespeare_char dataset and exploring checkpoints.

set -euo pipefail

DATA_DIR="data/shakespeare_char"
OUT_DIR="out_shakespeare_checkpoint_demo"
HIST_DIR="${OUT_DIR}/regex_histograms"
CKPT_PATH="${OUT_DIR}/ckpt.pt"

# Ensure the dataset is available.
pushd "${DATA_DIR}" > /dev/null
bash get_dataset.sh
popd > /dev/null

# Train a compact model to produce a checkpoint for analysis.
rm -rf "${OUT_DIR}"
python3 train.py \
  --dataset shakespeare_char \
  --out_dir "${OUT_DIR}" \
  --max_iters 200 \
  --block_size 64 \
  --batch_size 64 \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 128 \
  --learning_rate 3e-4 \
  --eval_interval 100 \
  --eval_iters 20 \
  --log_interval 10 \
  --always_save_checkpoint \
  --no-tensorboard_log

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Expected checkpoint not found at ${CKPT_PATH}" >&2
  exit 1
fi

# Use the interactive explorer to inspect embedding statistics.
echo "\nRunning checkpoint explorer to inspect transformer.wte.weight statistics..."
python3 analysis/checkpoint_analysis/checkpoint_explorer.py "${CKPT_PATH}" <<'EOINPUT'
0
0
0
4

b
b
b
q
EOINPUT

# Use the regex explorer to summarize attention projection weights and save histograms.
echo "\nRunning checkpoint regex explorer for attention projection weights..."
python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
  "${CKPT_PATH}" \
  "transformer\\.h\\.[0-9]+\\.attn\\.(c_attn|c_proj)\\.weight" \
  --max-rows 8 \
  --max-l2-rows 8 \
  --histogram-dir "${HIST_DIR}" \
  --histogram-bins 40

echo "\nHistogram images saved under ${HIST_DIR}"
