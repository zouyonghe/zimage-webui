# Z-Image WebUI

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-CUDA%20GPU-green.svg)](https://developer.nvidia.com/cuda-zone)

[中文](README.md) | [English](README_en.md) | [日本語](README_jp.md)

**基于本地权重的轻量级 AI 图像生成 Web 界面**

</div>

## 📖 项目简介

Z-Image WebUI 是一个基于本地 AI 模型的轻量级图像生成界面，提供直观的 Web 操作体验。无需网络连接，完全在本地运行，保护您的创作隐私。

### ✨ 核心特性

- 🎨 **直观的 Web 界面** - 基于 Vue 3 的现代化单页应用
- 🌍 **多语言支持** - 内置中文、英文、日文界面切换
- 🖼️ **智能画幅预设** - 支持常见分辨率比例，自动对齐到 16 像素步长
- ⚡ **批量生成** - 一次点击生成 1-10 张图片，支持随机或固定种子
- 💾 **自动保存** - 生成结果自动保存到本地，包含完整元数据
- 🔍 **高清放大** - 内置 Real-ESRGAN 超分辨率技术，支持 1-4 倍放大
- 🔎 **放大镜功能** - 预览时提供细节查看，节省性能
- 🎯 **即开即用** - 首次加载自动填充示例提示词

## 🚀 快速开始

### 环境要求

- **Python**: 3.10 或更高版本
- **GPU**: 支持 CUDA 的 NVIDIA 显卡
- **内存**: 建议 8GB 以上显存
- **系统**: Linux / Windows / macOS

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/zouyonghe/zimage-webui.git
   cd zimage-webui
   ```

2. **安装 PyTorch**（根据您的 CUDA 版本选择）
   ```bash
   # CUDA 12.1 示例
   pip install torch==2.5.1+cu121 torchvision==0.20.1 -f https://download.pytorch.org/whl/torch_stable.html
   ```

3. **安装项目依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **下载模型权重**
   ```bash
   cd scripts && bash download_models.sh && cd ..
   ```

### 启动服务

```bash
python webui_server.py
```

服务默认运行在 `http://localhost:9000`，您可以通过环境变量 `ZIMAGE_PORT` 修改端口。

### 命令行使用

```bash
# 使用默认提示词
python zimage.py

# 使用自定义提示词
python zimage.py "a scenic mountain landscape"
```

## 📁 项目结构

```
zimage-webui/
├── webui/                    # 前端资源
│   ├── index.html           # 主页面（Vue 3 SPA）
│   └── favicon-*.png        # 图标文件
├── webui_server.py          # Web 服务器和 API
├── zimage.py               # 命令行工具
├── zimage-model/           # AI 模型权重目录
├── weights/                # 超分模型权重
├── outputs/                # 生成结果保存目录
├── scripts/                # 辅助脚本
│   └── download_models.sh  # 模型下载脚本
└── requirements.txt        # Python 依赖包
```

## 🎯 功能详解

### 图像生成参数

| 参数 | 说明 | 范围 |
|------|------|------|
| 提示词 | 描述希望生成的内容 | 任意文本 |
| 负面词 | 描述不希望出现的内容 | 任意文本 |
| 生成步数 | 控制生成质量 | 1-50 |
| 引导强度 | 控制对提示词的遵循程度 | 1.0-20.0 |
| 随机种子 | 控制生成的随机性 | 任意整数或留空 |

### 画幅预设

- **方形**: 512×512, 768×768, 1024×1024
- **横向**: 768×512, 1024×768, 1024×576
- **纵向**: 512×768, 768×1024, 576×1024
- **宽屏**: 1024×512, 1152×648
- **竖屏**: 512×1024, 648×1152

### 高清放大

- **放大倍数**: 1x, 2x, 3x, 4x
- **模型**: Real-ESRGAN_x4plus
- **自动放大**: 可选择对新生成的图片自动应用超分
- **质量优化**: 保持细节的同时提升分辨率

## ⚙️ 配置说明

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `ZIMAGE_PORT` | 9000 | Web 服务端口 |
| `ZIMAGE_UPSCALE_MODEL` | weights/RealESRGAN_x4plus.pth | 超分模型路径 |

### 自定义配置

修改分辨率限制（需同时修改前后端）：
- 前端配置：`webui/index.html`
- 后端配置：`webui_server.py`

默认限制：
- 最小分辨率：512×512
- 最大分辨率：1024×1024
- 步长：16 像素

## 🔧 故障排除

### 常见问题

**Q: 提示 CUDA 不可用**
```bash
# 检查 CUDA 安装
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Q: Real-ESRGAN 安装警告**
```bash
# 可忽略警告，或安装 tensorboard 消除警告
pip install tensorboard
pip install realesrgan --no-deps
```

**Q: 显存不足**
- 减小生成分辨率
- 降低批量生成数量
- 关闭自动放大功能

**Q: 模型下载失败**
```bash
# 检查 aria2c 是否安装
aria2c --version

# 手动下载模型到对应目录
```

### 性能优化

1. **显存优化**
   - 启用 xformers 内存高效注意力
   - 使用可扩展显存模式
   - 启用注意力切片

2. **生成速度优化**
   - 使用适当的精度（BF16/FP16）
   - 调整生成步数
   - 合理设置批量大小

## 📊 生成结果

### 文件命名格式

生成的图片自动保存到 `outputs/` 目录，文件名格式：
```
{时间戳}_{宽度}x{高度}_{种子}.png
```

示例：`20240614_153045_768x768_rand.png`

### 元数据信息

每张图片包含完整的生成参数：
- 提示词和负面词
- 生成参数（步数、引导强度、种子）
- 生成时间戳
- 放大倍数（如适用）

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境设置

```bash
# 克隆项目
git clone https://github.com/zouyonghe/zimage-webui.git
cd zimage-webui

# 安装开发依赖
pip install -r requirements.txt

# 运行测试
python scripts/test_cuda.py
```

### 代码规范

- 遵循 PEP 8 Python 代码规范
- 使用语义化的 Git 提交信息
- 为新功能添加适当的文档

## 📄 许可证

本项目基于 [MIT 许可证](LICENSE) 发布。

## 🙏 致谢

- [Diffusers](https://github.com/huggingface/diffusers) - 强大的扩散模型库
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - 优秀的超分辨率模型
- [Vue.js](https://vuejs.org/) - 现代化的前端框架
- [Element Plus](https://element-plus.org/) - 优秀的 Vue 3 组件库

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [GitHub Issue](https://github.com/zouyonghe/zimage-webui/issues)
- 项目主页：[https://github.com/zouyonghe/zimage-webui](https://github.com/zouyonghe/zimage-webui)

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个 Star！**

</div>