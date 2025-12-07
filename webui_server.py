import base64
import io
import json
import os
import signal
import sys
import contextlib
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlparse

import numpy as np
import torch
from diffusers import ZImagePipeline
from PIL import Image
try:
    # 兼容 torchvision>=0.15 移除 functional_tensor 的情况
    import torchvision.transforms.functional_tensor as _tv_ft  # type: ignore
except Exception:  # noqa: BLE001
    try:
        import torchvision.transforms._functional_tensor as _tv_ft  # type: ignore
        sys.modules["torchvision.transforms.functional_tensor"] = _tv_ft
    except Exception:  # noqa: BLE001
        _tv_ft = None
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except Exception as exc:  # noqa: BLE001
    RealESRGANer = None
    RRDBNet = None
    _UPSCALE_IMPORT_ERROR = str(exc)
else:
    _UPSCALE_IMPORT_ERROR = ""

# ============================
# 显存优化设置
# ============================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).parent
WEB_DIR = ROOT / "webui"
DEFAULT_PROMPT = "a cat sitting on a chair, high quality, detailed"
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
HOST = "0.0.0.0"
PORT = int(os.environ.get("ZIMAGE_PORT", 9000))
CPU_OFFLOAD = os.environ.get("ZIMAGE_CPU_OFFLOAD", "").lower() in {"1", "true", "yes", "on"}
MAX_RESOLUTION = 1024
MIN_RESOLUTION = 512
RESOLUTION_STEP = 16
MAX_STEPS = 50
MAX_GUIDANCE = 20.0
MIN_GUIDANCE = 0.0
OUTPUT_DIR = ROOT / "outputs"
MAX_UPSCALE_FACTOR = 5.0
MAX_UPSCALE_EDGE = 4096
UPSCALE_MODEL_PATH = Path(os.environ.get("ZIMAGE_UPSCALE_MODEL", ROOT / "weights" / "RealESRGAN_x4plus.pth"))
UPSCALE_TILE = int(os.environ.get("ZIMAGE_UPSCALE_TILE", 256))
UPSCALE_TILE_PAD = int(os.environ.get("ZIMAGE_UPSCALE_TILE_PAD", 10))


_PIPE = None
_DEVICE = None
_DTYPE = None
_PIPE_LOCK = Lock()
_GEN_LOCK = Lock()
_WARMING = False
_PIPE_ERROR = None
_UPSCALER = None
_UPSCALER_LOCK = Lock()
_UPSCALER_ERROR = None


def get_pipeline():
    global _PIPE, _DEVICE, _DTYPE, _PIPE_ERROR  # noqa: PLW0603

    if _PIPE is not None:
        return _PIPE, _DEVICE, _DTYPE

    with _PIPE_LOCK:
        if _PIPE is not None:
            return _PIPE, _DEVICE, _DTYPE

        if not torch.cuda.is_available():
            _PIPE_ERROR = "CUDA unavailable"
            raise RuntimeError("CUDA unavailable")

        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        print("Loading Z-Image-Turbo from local weights...")
        try:
            pipe = ZImagePipeline.from_pretrained(
                str(ROOT / "zimage-model"),
                torch_dtype=dtype,
                local_files_only=True,
            )
            if CPU_OFFLOAD:
                pipe.enable_model_cpu_offload()
                print("Enabled model CPU offload.")
            else:
                pipe = pipe.to(device)
        except Exception as exc:  # noqa: BLE001
            _PIPE_ERROR = str(exc)
            raise

        # xformers：显存 -20～40%
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention.")
        except Exception as exc:  # noqa: BLE001
            print("xformers not available:", exc)

        pipe.enable_attention_slicing()
        print(f"Pipeline ready on {device} with dtype={dtype}.")

        _PIPE, _DEVICE, _DTYPE = pipe, device, dtype
        _PIPE_ERROR = None
        return _PIPE, _DEVICE, _DTYPE


def warmup_pipeline_async():
    global _WARMING  # noqa: PLW0603

    if _PIPE is not None or _WARMING:
        return

    _WARMING = True

    def _load():
        try:
            get_pipeline()
            print("Pipeline preloaded.")
        except Exception as exc:  # noqa: BLE001
            print(f"Pipeline preload failed: {exc}")
        finally:
            _WARMING = False

    Thread(target=_load, daemon=True).start()


def get_upscaler():
    global _UPSCALER, _UPSCALER_ERROR  # noqa: PLW0603

    if _UPSCALER is not None:
        return _UPSCALER

    if RealESRGANer is None or RRDBNet is None:
        raise RuntimeError(f"RealESRGAN not available: {_UPSCALE_IMPORT_ERROR}")

    with _UPSCALER_LOCK:
        if _UPSCALER is not None:
            return _UPSCALER

        model_path = UPSCALE_MODEL_PATH
        if not model_path.exists():
            _UPSCALER_ERROR = f"Upscale model not found at {model_path}"
            raise FileNotFoundError(_UPSCALER_ERROR)

        device = _DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        gpu_id = None if device == "cpu" else 0
        upsampler = RealESRGANer(
            scale=4,
            model_path=str(model_path),
            model=net,
            tile=UPSCALE_TILE,
            tile_pad=UPSCALE_TILE_PAD,
            pre_pad=0,
            half=device == "cuda",
            gpu_id=gpu_id,
        )
        _UPSCALER = (upsampler, device)
        _UPSCALER_ERROR = None
        print(f"Upscaler ready on {device} using model {model_path}")
        return _UPSCALER


class WebUIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _decode_base64_image(self, data_url: str):
        if not data_url:
            raise ValueError("image missing")
        if data_url.startswith("data:"):
            _, _, b64_part = data_url.partition(",")
            data_url = b64_part or data_url
        return base64.b64decode(data_url)

    def _send_json(self, status_code: int, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"

        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "cuda_available": torch.cuda.is_available(),
                    "pipeline_loaded": _PIPE is not None,
                    "pipeline_error": _PIPE_ERROR,
                    "pipeline_ready": (_PIPE is not None) and (_PIPE_ERROR is None),
                },
            )
            return

        if path == "/info":
            device = _DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = str(_DTYPE) if _DTYPE else ("torch.bfloat16" if device == "cuda" else "torch.float32")
            self._send_json(
                200,
                {
                    "model": "ZImagePipeline",
                    "device": device,
                    "dtype": dtype,
                    "defaults": {
                        "prompt": DEFAULT_PROMPT,
                        "steps": 9,
                        "guidance": 0.0,
                        "height": 512,
                        "width": 512,
                    },
                },
            )
            return

        if path == "/warmup":
            try:
                pipe, device, dtype = get_pipeline()
                self._send_json(200, {"status": "ready", "device": device, "dtype": str(dtype)})
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": f"Warmup failed: {exc}"})
            return

        super().do_GET()

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            content_length = int(self.headers.get("content-length", "0"))
            payload_raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            payload = json.loads(payload_raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": f"Invalid request body: {exc}"})
            return

        if path == "/upscale":
            return self._handle_upscale(payload)
        if path != "/generate":
            self.send_error(404, "Unsupported endpoint")
            return

        try:
            prompt = (payload.get("prompt") or "").strip() or DEFAULT_PROMPT
            negative_prompt = (payload.get("negative_prompt") or "").strip()
            steps = int(payload.get("steps", 9))
            guidance = float(payload.get("guidance", 0.0))
            height = int(payload.get("height", DEFAULT_HEIGHT))
            width = int(payload.get("width", DEFAULT_WIDTH))
            seed = payload.get("seed")
            if seed is not None:
                seed = int(seed)
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": f"Invalid parameter type: {exc}"})
            return

        # clamp and snap dimensions; if below min, fall back to defaults
        if height < MIN_RESOLUTION:
            height = DEFAULT_HEIGHT
        if width < MIN_RESOLUTION:
            width = DEFAULT_WIDTH
        height = max(MIN_RESOLUTION, min(height, MAX_RESOLUTION))
        width = max(MIN_RESOLUTION, min(width, MAX_RESOLUTION))
        height = (height // RESOLUTION_STEP) * RESOLUTION_STEP
        width = (width // RESOLUTION_STEP) * RESOLUTION_STEP

        if steps < 1 or steps > MAX_STEPS:
            self._send_json(400, {"error": f"steps must be between 1 and {MAX_STEPS}"})
            return

        if guidance < MIN_GUIDANCE or guidance > MAX_GUIDANCE:
            self._send_json(400, {"error": f"guidance must be between {MIN_GUIDANCE} and {MAX_GUIDANCE}"})
            return

        try:
            pipe, device, dtype = get_pipeline()
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": f"Pipeline init failed: {exc}"})
            return

        generator = torch.Generator(device=device)
        if seed is not None:
            try:
                generator = generator.manual_seed(int(seed))
            except Exception:  # noqa: BLE001
                self._send_json(400, {"error": "Seed must be an integer"})
                return
        else:
            seed = torch.seed()
            generator = generator.manual_seed(int(seed))

        print(
            f"Generating image | prompt='{prompt}' steps={steps} guidance={guidance} size={width}x{height} seed={seed} device={device}"
        )

        try:
            with _GEN_LOCK:
                with torch.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else torch.no_grad():
                    result = pipe(
                        prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        height=height,
                        width=width,
                        negative_prompt=negative_prompt or None,
                        generator=generator,
                    )
            image = result.images[0]
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": f"Generation failed: {exc}"})
            return

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        # 保存到本地文件夹，带时间戳
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_seed = seed if seed is not None else "rand"
        file_path = OUTPUT_DIR / f"{timestamp}_{width}x{height}_{filename_seed}.png"
        try:
            image.save(file_path, format="PNG")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to save image to disk: {exc}")
            file_path = None

        response = {
            "image": f"data:image/png;base64,{encoded}",
            "meta": {
                "prompt": prompt,
                "steps": steps,
                "guidance": guidance,
                "height": height,
                "width": width,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "device": device,
                "dtype": str(dtype),
                "saved_path": str(file_path) if file_path else None,
            },
        }
        self._send_json(200, response)

    def _handle_upscale(self, payload: dict):
        try:
            image_b64 = payload.get("image")
            scale = float(payload.get("scale", 2.0))
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": f"Invalid parameter type: {exc}"})
            return

        if not image_b64:
            self._send_json(400, {"error": "image is required"})
            return

        if scale <= 0:
            self._send_json(400, {"error": "scale must be positive"})
            return

        scale = max(1.0, min(scale, MAX_UPSCALE_FACTOR))

        try:
            decoded = self._decode_base64_image(image_b64)
            image = Image.open(io.BytesIO(decoded)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": f"Invalid image data: {exc}"})
            return

        src_w, src_h = image.size
        target_w = int(src_w * scale)
        target_h = int(src_h * scale)

        if target_w > MAX_UPSCALE_EDGE or target_h > MAX_UPSCALE_EDGE:
            aspect = src_w / src_h
            if aspect >= 1:
                target_w = MAX_UPSCALE_EDGE
                target_h = int(target_w / aspect)
            else:
                target_h = MAX_UPSCALE_EDGE
                target_w = int(target_h * aspect)
            scale = round(target_w / src_w, 2)

        try:
            upscaler, device = get_upscaler()
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": f"Upscale unavailable: {exc}"})
            return

        try:
            # RealESRGAN expects BGR numpy input; silence verbose tile logs
            img_np = np.array(image)[:, :, ::-1]
            with contextlib.redirect_stdout(io.StringIO()):
                output, _ = upscaler.enhance(img_np, outscale=scale)
            upscaled = Image.fromarray(output[:, :, ::-1])
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": f"Upscale failed: {exc}"})
            return

        buffer = io.BytesIO()
        upscaled.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = OUTPUT_DIR / f"upscaled_{timestamp}_{target_w}x{target_h}.png"
        try:
            upscaled.save(file_path, format="PNG")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to save upscaled image: {exc}")
            file_path = None

        self._send_json(
            200,
            {
                "image": f"data:image/png;base64,{encoded}",
                "meta": {
                    "type": "upscale",
                    "source_width": src_w,
                    "source_height": src_h,
                    "width": target_w,
                    "height": target_h,
                    "applied_scale": scale,
                    "saved_path": str(file_path) if file_path else None,
                },
            },
        )

    def log_message(self, fmt, *args):  # noqa: D401,N802
        """Silence noisy health polling logs."""
        if getattr(self, "path", "").startswith("/health"):
            return
        return super().log_message(fmt, *args)


def run_server():
    if not WEB_DIR.exists():
        print(f"Static directory not found: {WEB_DIR}")
        return

    warmup_pipeline_async()

    server = ThreadingHTTPServer((HOST, PORT), WebUIHandler)
    server.daemon_threads = True  # allow Ctrl+C to exit even if requests are running
    print(f"Serving WebUI on http://{HOST}:{PORT}")
    print("模型在后台预加载中，WebUI 已立即可用。")
    print("Press Ctrl+C to stop.")
    should_stop = False

    def handle_sigint(signum, frame):  # noqa: ANN001
        nonlocal should_stop
        if should_stop:
            print("Force exiting.")
            os._exit(1)  # noqa: PLR1722
        should_stop = True
        print("\nShutting down...")
        Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    server.serve_forever()
    server.server_close()


if __name__ == "__main__":
    run_server()
