import os
from pathlib import Path

import torch
from diffusers import ZImagePipeline

# ============================
# 显存优化设置（强烈推荐）
# ============================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 如果显存碎片严重，开启可扩展显存模式
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_pipeline(model_dir: Path) -> ZImagePipeline:
    """Load the Z-Image pipeline with safe defaults."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; please run on a GPU machine.")

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading Z-Image-Turbo from {model_dir} with dtype={dtype} ...")
    pipe = ZImagePipeline.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,  # 4090 支持 BF16，非常稳定
        local_files_only=True,
    ).to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Enabled xformers memory efficient attention.")
    except Exception as exc:  # noqa: BLE001
        print("xformers not available:", exc)

    pipe.enable_attention_slicing()
    return pipe


def main():
    model_dir = Path("./zimage-model")
    output_path = Path("zimage_test.png")
    prompt = "a cat sitting on a chair, high quality, detailed"

    try:
        pipe = load_pipeline(model_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load pipeline: {exc}")
        return

    print("Generating...")
    image = pipe(
        prompt,
        num_inference_steps=9,
        guidance_scale=0.0,
        height=512,
        width=512,
    ).images[0]
    image.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
