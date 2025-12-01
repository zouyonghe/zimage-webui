import base64
import io
import json
import os
import signal
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlparse

import torch
from diffusers import ZImagePipeline

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


_PIPE = None
_DEVICE = None
_DTYPE = None
_PIPE_LOCK = Lock()
_GEN_LOCK = Lock()
_WARMING = False
_PIPE_ERROR = None


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


class WebUIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

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
        if path != "/generate":
            self.send_error(404, "Unsupported endpoint")
            return

        try:
            content_length = int(self.headers.get("content-length", "0"))
            payload_raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            payload = json.loads(payload_raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": f"Invalid request body: {exc}"})
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
