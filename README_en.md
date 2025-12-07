# Z-Image WebUI

Lightweight local Web UI for Z-Image with aspect presets, batch generation, auto-save, hi-res upscale, and magnifier.

## Features
- Vue SPA with prompt/negative, steps, guidance, seed.
- Aspect presets (512/768/1024 square + common ratios), inputs snap to 16px, clamped 512–1024.
- Batch generate 1–10 images; empty seed = random per image, fixed seed = reproducible.
- Auto-save to `outputs/` with timestamp/size/seed; `meta.saved_path` returned by API.
- Hi-res upscale via Real-ESRGAN (1–4x), optional auto-upscale toggle; upscale results and metadata recorded.
- Magnifier toggle in preview (off by default to save performance).
- UI language switch: zh / en / ja.

## Requirements
- Python 3.10+
- CUDA GPU with matching PyTorch build
- Z-Image weights under `zimage-model/` (local only)
- Optional: Real-ESRGAN weight `weights/RealESRGAN_x4plus.pth` (env `ZIMAGE_UPSCALE_MODEL` to override)

## Install
```bash
# Install torch/torchvision matching your CUDA (example CUDA 12.1)
# pip install torch==2.5.1+cu121 torchvision==0.20.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

# Download model weights (aria2c required)
cd scripts && bash download_models.sh && cd ..
```

## Run
```bash
python webui_server.py
# Default 0.0.0.0:9000, override with ZIMAGE_PORT
```

Open `http://localhost:9000` in your browser.

Outputs are saved to `outputs/` like `20240614_153045_768x768_rand.png`.

## Notes
- Resolution limits: min 512 / max 1024 / step 16; adjust both `webui/index.html` and `webui_server.py` if you change them.
- Real-ESRGAN install warning about `tb-nightly` can be ignored; to silence, install `tensorboard` then `pip install realesrgan --no-deps`. Only requirement is your runtime env can `import realesrgan`.
