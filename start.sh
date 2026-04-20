#!/usr/bin/env sh
set -e

export PYTHONPATH="src"
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT="${PORT:-7860}"
export CONTEXT_DIR="${CONTEXT_DIR:-/data/contexts}"
export VECTOR_STORE_PATH="${VECTOR_STORE_PATH:-/data/contexts/faiss.index}"
export CHUNKS_PATH="${CHUNKS_PATH:-/data/contexts/chunks.json}"
export FINETUNE_DATA_PATH="${FINETUNE_DATA_PATH:-/data/finetune/train.jsonl}"
export OUTPUT_MODEL_DIR="${OUTPUT_MODEL_DIR:-/data/finetune/adapters}"
export AUTH_FILE_PATH="${AUTH_FILE_PATH:-/data/auth.json}"
export HF_HOME="${HF_HOME:-/data/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/data/hf-cache/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/hf-cache/datasets}"

# Em deploy novo no Railway, deixe OFFLINE_MODE=0 para permitir baixar modelos.
# Depois de cachear, voce pode mudar para OFFLINE_MODE=1.
export OFFLINE_MODE="${OFFLINE_MODE:-0}"

mkdir -p "$CONTEXT_DIR" "$(dirname "$VECTOR_STORE_PATH")" "$(dirname "$CHUNKS_PATH")"
mkdir -p "$(dirname "$FINETUNE_DATA_PATH")" "$OUTPUT_MODEL_DIR" "$(dirname "$AUTH_FILE_PATH")"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE"

python main.py