# Z-Image WebUI

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-CUDA%20GPU-green.svg)](https://developer.nvidia.com/cuda-zone)

[中文](README.md) | [English](README_en.md) | [日本語](README_jp.md)

**Lightweight AI Image Generation Web Interface with Local Models**

</div>

## 📖 Project Overview

Z-Image WebUI is a lightweight image generation interface based on local AI models, providing an intuitive web operation experience. No internet connection required, runs completely locally to protect your creative privacy.

### ✨ Core Features

- 🎨 **Intuitive Web Interface** - Modern single-page application based on Vue 3
- 🌍 **Multi-language Support** - Built-in Chinese, English, and Japanese interface switching
- 🖼️ **Smart Aspect Presets** - Support for common resolution ratios, auto-aligned to 16-pixel steps
- ⚡ **Batch Generation** - Generate 1-10 images with one click, support random or fixed seeds
- 💾 **Auto-save** - Generation results automatically saved locally with complete metadata
- 🔍 **HD Upscaling** - Built-in Real-ESRGAN super-resolution technology, support 1-4x magnification
- 🔎 **Magnifier Feature** - Detail viewing during preview, saves performance
- 🎯 **Ready to Use** - Auto-fill example prompts on first load

## 🚀 Quick Start

### Requirements

- **Python**: 3.10 or higher
- **GPU**: CUDA-enabled NVIDIA graphics card
- **Memory**: Recommended 8GB+ VRAM
- **System**: Linux / Windows / macOS

### Installation Steps

1. **Clone Project**
   ```bash
   git clone https://github.com/zouyonghe/zimage-webui.git
   cd zimage-webui
   ```

2. **Install PyTorch** (Choose according to your CUDA version)
   ```bash
   # CUDA 12.1 example
   pip install torch==2.5.1+cu121 torchvision==0.20.1 -f https://download.pytorch.org/whl/torch_stable.html
   ```

3. **Install Project Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Model Weights**
   ```bash
   cd scripts && bash download_models.sh && cd ..
   ```

### Start Service

```bash
python webui_server.py
```

Service runs on `http://localhost:9000` by default, you can change the port via environment variable `ZIMAGE_PORT`.

### Command Line Usage

```bash
# Use default prompt
python zimage.py

# Use custom prompt
python zimage.py "a scenic mountain landscape"
```

## 📁 Project Structure

```
zimage-webui/
├── webui/                    # Frontend resources
│   ├── index.html           # Main page (Vue 3 SPA)
│   └── favicon-*.png        # Icon files
├── webui_server.py          # Web server and API
├── zimage.py               # Command line tool
├── zimage-model/           # AI model weights directory
├── weights/                # Upscaling model weights
├── outputs/                # Generation results save directory
├── scripts/                # Helper scripts
│   └── download_models.sh  # Model download script
└── requirements.txt        # Python dependencies
```

## 🎯 Feature Details

### Image Generation Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| Prompt | Describe desired content | Any text |
| Negative | Describe unwanted content | Any text |
| Steps | Control generation quality | 1-50 |
| Guidance | Control adherence to prompt | 1.0-20.0 |
| Seed | Control generation randomness | Any integer or leave empty |

### Aspect Presets

- **Square**: 512×512, 768×768, 1024×1024
- **Landscape**: 768×512, 1024×768, 1024×576
- **Portrait**: 512×768, 768×1024, 576×1024
- **Widescreen**: 1024×512, 1152×648
- **Vertical**: 512×1024, 648×1152

### HD Upscaling

- **Scale Factor**: 1x, 2x, 3x, 4x
- **Model**: Real-ESRGAN_x4plus
- **Auto Upscale**: Optional automatic upscaling for newly generated images
- **Quality Optimization**: Enhance resolution while preserving details

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ZIMAGE_PORT` | 9000 | Web service port |
| `ZIMAGE_UPSCALE_MODEL` | weights/RealESRGAN_x4plus.pth | Upscaling model path |

### Custom Configuration

To modify resolution limits (need to modify both frontend and backend):
- Frontend config: `webui/index.html`
- Backend config: `webui_server.py`

Default limits:
- Minimum resolution: 512×512
- Maximum resolution: 1024×1024
- Step: 16 pixels

## 🔧 Troubleshooting

### Common Issues

**Q: CUDA not available**
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Q: Real-ESRGAN installation warning**
```bash
# Can ignore warning, or install tensorboard to eliminate warning
pip install tensorboard
pip install realesrgan --no-deps
```

**Q: Insufficient VRAM**
- Reduce generation resolution
- Lower batch generation count
- Disable auto upscaling

**Q: Model download failed**
```bash
# Check if aria2c is installed
aria2c --version

# Manually download models to corresponding directories
```

### Performance Optimization

1. **Memory Optimization**
   - Enable xformers memory efficient attention
   - Use expandable memory mode
   - Enable attention slicing

2. **Generation Speed Optimization**
   - Use appropriate precision (BF16/FP16)
   - Adjust generation steps
   - Set reasonable batch size

## 📊 Generation Results

### File Naming Format

Generated images are automatically saved to `outputs/` directory with filename format:
```
{timestamp}_{width}x{height}_{seed}.png
```

Example: `20240614_153045_768x768_rand.png`

### Metadata Information

Each image contains complete generation parameters:
- Prompt and negative prompt
- Generation parameters (steps, guidance, seed)
- Generation timestamp
- Upscale factor (if applicable)

## 🤝 Contributing

Welcome to submit Issues and Pull Requests!

### Development Environment Setup

```bash
# Clone project
git clone https://github.com/zouyonghe/zimage-webui.git
cd zimage-webui

# Install development dependencies
pip install -r requirements.txt

# Run tests
python scripts/test_cuda.py
```

### Code Standards

- Follow PEP 8 Python code standards
- Use semantic Git commit messages
- Add appropriate documentation for new features

## 📄 License

This project is released under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Diffusers](https://github.com/huggingface/diffusers) - Powerful diffusion model library
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Excellent super-resolution model
- [Vue.js](https://vuejs.org/) - Modern frontend framework
- [Element Plus](https://element-plus.org/) - Excellent Vue 3 component library

## 📞 Contact

For questions or suggestions, please contact us through:

- Submit [GitHub Issue](https://github.com/zouyonghe/zimage-webui/issues)
- Project homepage: [https://github.com/zouyonghe/zimage-webui](https://github.com/zouyonghe/zimage-webui)

---

<div align="center">

**⭐ If this project helps you, please give us a Star!**

</div>