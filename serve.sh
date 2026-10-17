#!/usr/bin/env bash
# Nemotron 3.5 Bolt 74B-A9B 24GB (IQ2 GGUF) + LiLiCorr drafter (k=7) on one DGX Spark / GB10.
# This is NVIDIA's published 128 GB ("32GB recipe") command. It measured fastest of every
# configuration we tried; see README for the arms that did not beat it.
set -euo pipefail
MODELS=${MODELS:-$HOME/models}
LLAMA=${LLAMA:-$HOME/src/llama.cpp}
TARGET_DIR=$MODELS/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF
DRAFT_DIR=$MODELS/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8090}

exec "$LLAMA/build/bin/llama-server" \
  --model "$TARGET_DIR/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf" \
  --mmproj "$TARGET_DIR/mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf" \
  --alias nemotron-3.5-bolt \
  --host "$HOST" --port "$PORT" \
  --batch-size 4096 \
  --ubatch-size 2048 \
  --ctx-size 262144 \
  --parallel 1 \
  -ngl 999 \
  --spec-type draft-dflash \
  --spec-draft-n-max 7 \
  --spec-draft-model "$DRAFT_DIR/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf" \
  --spec-draft-ngl 999 \
  --temp 1.0 \
  --top-p 0.95 \
  --flash-attn on \
  --jinja \
  ${EXTRA_ARGS:-}
