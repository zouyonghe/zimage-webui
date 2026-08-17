import argparse
import os
from pathlib import Path

import torch
from diffusers import ZImagePipeline

ROOT = Path(__file__).resolve().parent

# ============================
# 显存优化设置（强烈推荐）
# ============================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 如果显存碎片严重，开启可扩展显存模式
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEFAULT_PROMPT = "a cat sitting on a chair, high quality, detailed"
MODEL_CONFIGS = {
    "turbo": {
        "display_name": "Z-Image-Turbo",
        "path": ROOT / "zimage-model",
        "steps": 9,
        "guidance": 0.0,
    },
    "base": {
        "display_name": "Z-Image",
        "path": ROOT / "zimage-base-model",
        "steps": 50,
        "guidance": 4.0,
    },
}
CPU_OFFLOAD = os.environ.get("ZIMAGE_CPU_OFFLOAD", "").lower() in {"1", "true", "yes", "on"}


def load_pipeline(model_id: str, model_dir: Path) -> ZImagePipeline:
    """Load the Z-Image pipeline with safe defaults."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; please run on a GPU machine.")

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {MODEL_CONFIGS[model_id]['display_name']} ({model_id}) from {model_dir} with dtype={dtype} ...")
    pipe = ZImagePipeline.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,  # 4090 支持 BF16，非常稳定
        local_files_only=True,
    )

    if CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
        print("Enabled model CPU offload.")
    else:
        pipe = pipe.to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Enabled xformers memory efficient attention.")
    except Exception as exc:  # noqa: BLE001
        print("xformers not available:", exc)

    pipe.enable_attention_slicing()
    return pipe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an image with Z-Image.")
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_CONFIGS),
        default="turbo",
        help="Model variant to use (default: turbo)",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt to generate (optional; uses default prompt when omitted)",
    )
    return parser


def main():
    args = build_parser().parse_args()

    model_config = MODEL_CONFIGS[args.model]
    model_dir = model_config["path"]
    output_path = Path("zimage_test.png")

    prompt = " ".join(args.prompt).strip() if args.prompt else DEFAULT_PROMPT
    if not prompt:
        prompt = DEFAULT_PROMPT

    try:
        pipe = load_pipeline(args.model, model_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load pipeline: {exc}")
        return

    print(f"Generating with prompt: {prompt!r}")
    image = pipe(
        prompt,
        num_inference_steps=model_config["steps"],
        guidance_scale=model_config["guidance"],
        height=512,
        width=512,
        cfg_normalization=False,
    ).images[0]
    image.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
