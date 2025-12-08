# Z-Image WebUI

[中文](README.md) | [English](README_en.md) | [日本語](README_jp.md)

Lightweight local Web UI for Z-Image with aspect presets, batch generation, auto-save, hi-res upscale, and magnifier.

## Feature Overview
- Vue SPA with prompt/negative, steps, guidance, seed.
- Built-in language switch (Chinese / English / Japanese).
- Aspect presets (512/768/1024 square + common ratios); inputs snap to 16px and clamp to 512–1024.
- Batch generate 1–10 images; empty seed = random per image, fixed seed = reproducible.
- Auto-save to `outputs/` with timestamp/size/seed; API returns `meta.saved_path`.
- Preload a random prompt on first load for quick start.
- Hi-res upscale via Real-ESRGAN (1–4x); button in preview, optional “auto upscale” for subsequent generations; results and metadata recorded.
- Magnifier toggle in preview (off by default to save performance).

## Requirements
- Python 3.10+
- CUDA GPU (matching PyTorch build)
- Local Z-Image weights in `zimage-model/` (no network load)
- Optional: Real-ESRGAN weight `weights/RealESRGAN_x4plus.pth` (override with `ZIMAGE_UPSCALE_MODEL`)

## Install
```bash
# Install torch/torchvision matching CUDA (example CUDA 12.1)
# pip install torch==2.5.1+cu121 torchvision==0.20.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

# Download weights (aria2c required): main model to zimage-model/, RealESRGAN_x4plus to weights/
cd scripts && bash download_models.sh && cd ..
```

## Run
```bash
python webui_server.py
# Default 0.0.0.0:9000, override with ZIMAGE_PORT

# CLI quick run (local model)
python zimage.py                       # use default prompt
python zimage.py "a scenic mountain"   # custom prompt
```

Open `http://localhost:9000` in your browser.

Outputs save to `outputs/`, e.g. `20240614_153045_768x768_rand.png`.

## Directory
- `webui/`: front-end (Vue 3 ESM)
- `webui_server.py`: HTTP server & generation API
- `zimage-model/`: model weights
- `outputs/`: auto-saved images (created on run)
- Others: `requirements.txt`, helper/test scripts
- Download script: `scripts/download_models.sh` (uses hf-mirror)
- CUDA test: `scripts/test_cuda.py`

## Notes
- Resolution limits: min 512 / max 1024 / step 16; change both `webui/index.html` and `webui_server.py` if needed.
- Server logs include generation params/errors for debugging.
- Real-ESRGAN install warning about `tb-nightly` can be ignored; to silence, `pip install tensorboard` then `pip install realesrgan --no-deps`. Just ensure runtime can `import realesrgan`.
- Hi-res upscale results append to results/history with scale shown; “auto upscale” is off by default.
- Magnifier toggle lives in the preview toolbar; when off, magnifier is not rendered to save performance.

## License
Released under the MIT License. See `LICENSE` for details.
