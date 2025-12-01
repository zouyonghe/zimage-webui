import torch
from diffusers import ZImagePipeline

# ============================
# 显存优化设置（强烈推荐）
# ============================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 如果显存碎片严重，开启可扩展显存模式
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Loading Z-Image-Turbo from local weights...")

pipe = ZImagePipeline.from_pretrained(
    "./zimage-model",
    torch_dtype=torch.bfloat16,    # 4090 支持 BF16，非常稳定
    local_files_only=True,
)

# ============================
# 启用显存优化
# ============================
pipe = pipe.to("cuda")

# xformers：显存 -20～40%
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("Enabled xformers memory efficient attention.")
except Exception as e:
    print("xformers not available:", e)

# attention slicing：进一步降低峰值显存
pipe.enable_attention_slicing()

# ============================
# 生成图像
# ============================
print("Generating...")
image = pipe(
    "a cat sitting on a chair, high quality, detailed",
    num_inference_steps=9,
    guidance_scale=0.0,

    # 🚀 **关键：降低分辨率，防止 24GB 爆显存**
    height=512,
    width=512,
).images[0]

image.save("zimage_test.png")
print("Saved: zimage_test.png")
