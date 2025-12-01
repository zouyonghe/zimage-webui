#!/bin/bash

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

for f in "${files[@]}"; do
    mkdir -p "$(dirname "$f")"
    echo "Downloading $f..."
    aria2c -x 16 -s 16 -k 5M "$base/$f" -o "$f"
done
