# Z-Image WebUI

基于本地权重的轻量 Web UI，提供分辨率预设、连续批量生成和自动落盘保存等功能。

## 功能概览
- Vue 单页前端，支持提示词 / 负面词、步数、引导强度、种子。
- 内置中/英/日三语切换。
- 画幅预设下拉（包含 512/768/1024 的 1:1 及常见比例），输入框自动对齐到 16 像素步长并限制在 512–1024。
- 连续生成：一次点击可连续生成 1–10 张，默认每张使用独立随机种子；填入固定种子则按该值生成。
- 自动保存：每次生成自动写入 `outputs/`，文件名包含时间戳、分辨率和种子（无种子时标记为 `rand`）。接口返回的 `meta.saved_path` 也可查看保存路径。
- 初次加载随机填充一条提示词，便于直接试跑。
- 高分放大：内置 Real-ESRGAN 超分，参数区可调倍率（1–4x），预览区有“高分放大”按钮触发；可选“自动放大”开启后，对后续生成的图片自动超分。
- 放大镜：预览时可手动开启放大镜按钮查看局部细节（默认关闭，避免高分图性能开销）。

## 环境要求
- Python 3.10+
- CUDA GPU（需要可用的 PyTorch CUDA 版本）
- 本地 Z-Image 权重已放在 `zimage-model/`（默认使用本地权重，不走网络）
- （可选）Real-ESRGAN 权重用于模型超分：`weights/RealESRGAN_x4plus.pth`，可设置 `ZIMAGE_UPSCALE_MODEL` 覆盖路径

## 安装
```bash
# 安装匹配 CUDA 的 torch/torchvision（示例为 CUDA 12.1）
# pip install torch==2.5.1+cu121 torchvision==0.20.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

# 下载模型权重（需先安装 aria2c；脚本会下载主模型到 zimage-model/，并下载 RealESRGAN_x4plus 到 weights/）
cd scripts && bash download_models.sh && cd ..

```

## 运行
```bash
python webui_server.py
# 默认监听 0.0.0.0:9000，可通过环境变量 ZIMAGE_PORT 修改
```

浏览器打开 `http://localhost:9000`。

生成的图片会自动保存到 `outputs/`，文件名格式类似 `20240614_153045_768x768_rand.png`。

## 目录结构
- `webui/`：前端页面（Vue 3 ESM）
- `webui_server.py`：简易 HTTP 服务与生成接口
- `zimage-model/`：模型权重（默认从本地加载）
- `outputs/`：生成结果自动保存目录（运行后自动创建）
- 其他：`requirements.txt`、测试脚本等
- 模型下载脚本：`scripts/download_models.sh`（默认使用 hf-mirror 镜像）
- CUDA 测试脚本：`scripts/test_cuda.py`

## 其他
- 如果需要调整分辨率范围，前后端都有最小 512 / 最大 1024、步长 16 的限制；同步修改 `webui/index.html` 与 `webui_server.py` 中的参数即可。
- 服务器日志会打印生成请求参数与错误信息，便于排查。
- Real-ESRGAN 安装提示 `tb-nightly` 缺失可忽略；如需消除，可先 `pip install tensorboard`，再执行 `pip install realesrgan --no-deps`。只要启动服务的 Python 环境能 `import realesrgan` 即可使用模型超分。
- 高分放大结果会自动追加到结果列表与历史记录，历史中会显示放大倍数；默认“自动放大”关闭。
- 放大镜开关位于预览操作栏，仅在预览大图时使用，关闭后不会渲染放大镜以节省性能。
