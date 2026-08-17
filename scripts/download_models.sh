#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WEIGHTS_DIR="$ROOT_DIR/weights"
realesrgan_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
variant="${1:-all}"

cd "$ROOT_DIR"

command -v hf >/dev/null || {
    echo "Missing hf CLI. Install it with: pip install -U huggingface_hub" >&2
    exit 1
}

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

download_model() {
    local repo="$1"
    local target="$2"
    echo "Downloading $repo -> $target"
    hf download "$repo" --local-dir "$target" --max-workers "${HF_MAX_WORKERS:-1}"
}

case "$variant" in
    turbo)
        download_model "Tongyi-MAI/Z-Image-Turbo" "$ROOT_DIR/zimage-model"
        ;;
    base)
        download_model "Tongyi-MAI/Z-Image" "$ROOT_DIR/zimage-base-model"
        ;;
    all)
        download_model "Tongyi-MAI/Z-Image-Turbo" "$ROOT_DIR/zimage-model"
        download_model "Tongyi-MAI/Z-Image" "$ROOT_DIR/zimage-base-model"
        ;;
    *)
        echo "Usage: $0 [turbo|base|all]" >&2
        exit 2
        ;;
esac

# Download Real-ESRGAN weight
mkdir -p "$WEIGHTS_DIR"
echo "Downloading RealESRGAN_x4plus.pth -> $WEIGHTS_DIR/"
aria2c -x 16 -s 16 -k 5M "$realesrgan_url" -d "$WEIGHTS_DIR" -o "RealESRGAN_x4plus.pth"
