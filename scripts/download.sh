#!/bin/bash

base="https://hf-mirror.com/Tongyi-MAI/Z-Image-Turbo/resolve/main"

files=(
"text_encoder/model-00001-of-00003.safetensors"
"text_encoder/model-00002-of-00003.safetensors"
"text_encoder/model-00003-of-00003.safetensors"

"transformer/diffusion_pytorch_model-00001-of-00003.safetensors"
"transformer/diffusion_pytorch_model-00002-of-00003.safetensors"
"transformer/diffusion_pytorch_model-00003-of-00003.safetensors"

"vae/diffusion_pytorch_model.safetensors"
)

for f in "${files[@]}"; do
    mkdir -p "$(dirname "$f")"
    echo "Downloading $f..."
    aria2c -x 16 -s 16 -k 5M "$base/$f" -o "$f"
done
