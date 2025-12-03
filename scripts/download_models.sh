#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$ROOT_DIR/zimage-model"

base="https://hf-mirror.com/Tongyi-MAI/Z-Image-Turbo/resolve/main"

files=(
"text_encoder/model-00001-of-00003.safetensors"  # text encoder shard 1/3
"text_encoder/model-00002-of-00003.safetensors"  # text encoder shard 2/3
"text_encoder/model-00003-of-00003.safetensors"  # text encoder shard 3/3

"transformer/diffusion_pytorch_model-00001-of-00003.safetensors"  # transformer shard 1/3
"transformer/diffusion_pytorch_model-00002-of-00003.safetensors"  # transformer shard 2/3
"transformer/diffusion_pytorch_model-00003-of-00003.safetensors"  # transformer shard 3/3

"vae/diffusion_pytorch_model.safetensors"  # VAE
)

cd "$ROOT_DIR"

for f in "${files[@]}"; do
    target_dir="$MODEL_DIR/$(dirname "$f")"
    target_file="$(basename "$f")"
    mkdir -p "$target_dir"
    echo "Downloading $f -> $MODEL_DIR/$f"
    aria2c -x 16 -s 16 -k 5M "$base/$f" -d "$target_dir" -o "$target_file"
done
