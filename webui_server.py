import base64
import gc
import io
import json
import math
import os
import re
import signal
import socket
import sys
import time
import contextlib
import inspect
import uuid
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread, Semaphore
from urllib.parse import urlparse
from typing import Optional, Tuple

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
MODEL_REGISTRY = {
    "turbo": {
        "display_name": "Z-Image-Turbo",
        "path": ROOT / "zimage-model",
        "defaults": {"steps": 9, "guidance": 0.0, "cfg_normalization": False},
    },
    "base": {
        "display_name": "Z-Image",
        "path": ROOT / "zimage-base-model",
        "defaults": {"steps": 50, "guidance": 4.0, "cfg_normalization": False},
    },
}
REQUIRED_MODEL_PATHS = ("model_index.json", "text_encoder", "tokenizer", "transformer", "vae")
DEFAULT_PROMPT = "a cat sitting on a chair, high quality, detailed"
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
HOST = os.environ.get("ZIMAGE_HOST", "127.0.0.1")
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
UPSCALE_MAX_CONCURRENCY = int(os.environ.get("ZIMAGE_UPSCALE_CONCURRENCY", 1))
UPSCALE_WAIT_TIMEOUT = int(os.environ.get("ZIMAGE_UPSCALE_WAIT_TIMEOUT", 60))  # seconds
SSE_WRITE_TIMEOUT = float(os.environ.get("ZIMAGE_SSE_WRITE_TIMEOUT", 5.0))
UPSCALE_DEFAULT_FORMAT = os.environ.get("ZIMAGE_UPSCALE_FORMAT", "png").lower()


def clamp_resolution(height: int, width: int) -> Tuple[int, int]:
    """Clamp resolution to configured bounds and grid."""
    h = int(height)
    w = int(width)
    if h < MIN_RESOLUTION:
        h = DEFAULT_HEIGHT
    if w < MIN_RESOLUTION:
        w = DEFAULT_WIDTH
    h = max(MIN_RESOLUTION, min(h, MAX_RESOLUTION))
    w = max(MIN_RESOLUTION, min(w, MAX_RESOLUTION))
    h = (h // RESOLUTION_STEP) * RESOLUTION_STEP
    w = (w // RESOLUTION_STEP) * RESOLUTION_STEP
    return h, w


def build_generator(device: str, seed: Optional[int]) -> Tuple[torch.Generator, int]:
    """Create a torch.Generator seeded consistently; returns generator and resolved seed."""
    gen = torch.Generator(device=device)
    resolved = int(seed) if seed is not None else int(torch.seed())
    gen = gen.manual_seed(resolved)
    return gen, resolved


def generation_context(device: str, dtype: torch.dtype):
    return torch.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else torch.no_grad()


_PIPE = None
_DEVICE = None
_DTYPE = None
_ACTIVE_MODEL_ID = None
_MODEL_LOADING = False
_PIPE_ERROR = None
_RUNTIME_LOCK = Lock()
_STATE_LOCK = Lock()
_UPSCALER = None
_UPSCALER_LOCK = Lock()
_UPSCALER_ERROR = None
_UPSCALE_SEMAPHORE = Semaphore(max(1, UPSCALE_MAX_CONCURRENCY))


def _new_request_id() -> str:
    return uuid.uuid4().hex[:8]


def _save_image_async(image: Image.Image, path: Path, *, fmt: str = "png", compress_level: int = 1, quality: int = 92) -> None:
    """Save image asynchronously to avoid blocking the response."""

    def _worker():
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            kwargs = {}
            fmt_local = fmt.upper()
            if fmt_local == "PNG":
                kwargs["compress_level"] = compress_level
            elif fmt_local in {"JPEG", "WEBP"}:
                kwargs["quality"] = quality
            image.save(path, format=fmt_local, **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to save upscaled image: {exc}")

    Thread(target=_worker, daemon=True).start()


class UnknownModelError(ValueError):
    pass


class ModelUnavailableError(RuntimeError):
    pass


class RuntimeBusyError(RuntimeError):
    pass


class NoActiveModelError(RuntimeError):
    pass


def _missing_model_paths(model_path: Path) -> list[str]:
    missing = []
    for relative_path in REQUIRED_MODEL_PATHS:
        path = model_path / relative_path
        exists = path.is_file() if relative_path == "model_index.json" else path.is_dir()
        if not exists:
            missing.append(relative_path)
    return missing


def _runtime_state_snapshot() -> tuple:
    with _STATE_LOCK:
        return _ACTIVE_MODEL_ID, _MODEL_LOADING, _PIPE_ERROR, _DEVICE, _DTYPE


def _publish_runtime_state(*, active_model=None, loading=False, error=None, device=None, dtype=None) -> None:
    global _ACTIVE_MODEL_ID, _MODEL_LOADING, _PIPE_ERROR, _DEVICE, _DTYPE  # noqa: PLW0603

    with _STATE_LOCK:
        _ACTIVE_MODEL_ID = active_model
        _MODEL_LOADING = loading
        _PIPE_ERROR = error
        _DEVICE = device
        _DTYPE = dtype


def _set_runtime_loading(loading: bool) -> None:
    global _MODEL_LOADING  # noqa: PLW0603

    with _STATE_LOCK:
        _MODEL_LOADING = loading


def get_models_status() -> dict:
    active_model, loading, error, _device, _dtype = _runtime_state_snapshot()
    models = []
    for model_id, config in MODEL_REGISTRY.items():
        missing = _missing_model_paths(config["path"])
        models.append(
            {
                "id": model_id,
                "display_name": config["display_name"],
                "path": str(config["path"]),
                "available": not missing,
                "missing": missing,
                "defaults": dict(config["defaults"]),
            }
        )
    return {
        "models": models,
        "active_model": active_model,
        "loading": loading,
        "last_error": error,
    }


def get_active_pipeline():
    if _PIPE is None or _ACTIVE_MODEL_ID is None:
        raise NoActiveModelError("No model is active; load a model first")
    return _PIPE, _DEVICE, _DTYPE, MODEL_REGISTRY[_ACTIVE_MODEL_ID]


def _unload_active_pipeline(*, loading=False) -> None:
    global _PIPE  # noqa: PLW0603

    old_pipeline = _PIPE
    _PIPE = None
    _publish_runtime_state(loading=loading)
    if old_pipeline is not None:
        del old_pipeline
    gc.collect()
    torch.cuda.empty_cache()


def load_model(model_id: str) -> dict:
    global _PIPE  # noqa: PLW0603

    if not isinstance(model_id, str) or model_id not in MODEL_REGISTRY:
        raise UnknownModelError(f"Unknown model: {model_id}")

    if not _RUNTIME_LOCK.acquire(blocking=False):
        raise RuntimeBusyError("Runtime is busy")

    _set_runtime_loading(True)
    try:
        config = MODEL_REGISTRY[model_id]
        missing = _missing_model_paths(config["path"])
        if missing:
            raise ModelUnavailableError(f"Model '{model_id}' is incomplete; missing: {', '.join(missing)}")

        pipe = None
        try:
            _unload_active_pipeline(loading=True)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")

            device = "cuda"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"Loading {config['display_name']} from local weights...")
            pipe = ZImagePipeline.from_pretrained(
                str(config["path"]),
                torch_dtype=dtype,
                local_files_only=True,
            )
            if CPU_OFFLOAD:
                pipe.enable_model_cpu_offload()
                print("Enabled model CPU offload.")
            else:
                pipe = pipe.to(device)

            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("Enabled xformers memory efficient attention.")
            except Exception as exc:  # noqa: BLE001
                print("xformers not available:", exc)

            pipe.enable_attention_slicing()
            _PIPE = pipe
            _publish_runtime_state(active_model=model_id, device=device, dtype=dtype)
            print(f"Pipeline ready on {device} with dtype={dtype}.")
            return {"active_model": model_id, "defaults": dict(config["defaults"])}
        except Exception as exc:  # noqa: BLE001
            _PIPE = None
            _publish_runtime_state(error=str(exc))
            if pipe is not None:
                del pipe
            gc.collect()
            torch.cuda.empty_cache()
            raise
    finally:
        _set_runtime_loading(False)
        _RUNTIME_LOCK.release()


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


def _acquire_upscale_slot() -> bool:
    """Try to acquire an upscale concurrency slot."""
    return _UPSCALE_SEMAPHORE.acquire(blocking=False)


def _release_upscale_slot(acquired: bool):
    if acquired:
        _UPSCALE_SEMAPHORE.release()


def _select_upscale_tile(max_edge: int) -> Tuple[int, int]:
    """Select tile/pad based on target size to balance speed vs memory."""
    tile = UPSCALE_TILE
    if max_edge >= 3500:
        tile = min(tile, 128)
    elif max_edge >= 3000:
        tile = min(tile, 160)
    elif max_edge >= 2500:
        tile = min(tile, 192)
    elif max_edge >= 2000:
        tile = min(tile, 224)
    else:
        tile = min(tile, 256)
    pad = min(UPSCALE_TILE_PAD, max(4, tile // 8))
    return tile, pad


def _validate_image_format(fmt: str) -> str:
    allowed = {"png", "jpeg", "jpg", "webp"}
    fmt = (fmt or "png").lower()
    if fmt not in allowed:
        return "png"
    if fmt == "jpg":
        fmt = "jpeg"
    return fmt


def _encode_image(image: Image.Image, fmt: str) -> Tuple[str, str]:
    """Encode image to base64 string and return (data_url, mime)."""
    fmt = _validate_image_format(fmt)
    mime = "image/png" if fmt == "png" else f"image/{fmt}"
    params = {}
    if fmt in {"jpeg", "webp"}:
        params.update({"quality": 92, "optimize": False})
    if fmt == "png":
        params.update({"compress_level": 1})
    buffer = io.BytesIO()
    image.save(buffer, format=fmt.upper(), **params)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}", mime


def _wait_for_upscale_slot(timeout: int, sse_emit=None) -> bool:
    """Poll for an upscale slot with timeout; optionally emit SSE queue status."""
    step = 0.5
    waited = 0.0
    while waited < timeout:
        if _acquire_upscale_slot():
            return True
        waited += step
        if sse_emit:
            ok = sse_emit({"queued_for_seconds": round(waited, 1)})
            if not ok:
                return False
        time.sleep(step)
    return False


class WebUIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def handle(self):  # noqa: D401
        """Handle a single HTTP request; ignore client resets to avoid noisy traces."""
        try:
            super().handle()
        except ConnectionResetError:
            return

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

    def _parse_generate_params(self, payload: dict, defaults: dict) -> dict:
        try:
            prompt = (payload.get("prompt") or "").strip() or DEFAULT_PROMPT
            negative_prompt = (payload.get("negative_prompt") or "").strip()
            steps = int(payload.get("steps", defaults["steps"]))
            guidance = float(payload.get("guidance", defaults["guidance"]))
            height = int(payload.get("height", DEFAULT_HEIGHT))
            width = int(payload.get("width", DEFAULT_WIDTH))
            seed = payload.get("seed")
            if seed is not None:
                seed = int(seed)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid parameter type: {exc}") from exc

        height, width = clamp_resolution(height, width)

        if steps < 1 or steps > MAX_STEPS:
            raise ValueError(f"steps must be between 1 and {MAX_STEPS}")

        if guidance < MIN_GUIDANCE or guidance > MAX_GUIDANCE:
            raise ValueError(f"guidance must be between {MIN_GUIDANCE} and {MAX_GUIDANCE}")
        if not math.isfinite(guidance):
            raise ValueError("guidance must be finite")

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance": guidance,
            "height": height,
            "width": width,
            "seed": seed,
        }

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"

        if path == "/models":
            self._send_json(200, get_models_status())
            return

        if path == "/health":
            active_model, loading, error, _device, _dtype = _runtime_state_snapshot()
            self._send_json(
                200,
                {
                    "status": "ok",
                    "cuda_available": torch.cuda.is_available(),
                    "pipeline_loaded": active_model is not None,
                    "pipeline_error": error,
                    "pipeline_ready": active_model is not None,
                    "active_model": active_model,
                    "model_loading": loading,
                    "model_error": error,
                },
            )
            return

        if path == "/info":
            active_model, loading, error, device, dtype = _runtime_state_snapshot()
            model_config = MODEL_REGISTRY.get(active_model)
            defaults = None
            if model_config is not None:
                defaults = {
                    "prompt": DEFAULT_PROMPT,
                    **model_config["defaults"],
                    "height": DEFAULT_HEIGHT,
                    "width": DEFAULT_WIDTH,
                }
            self._send_json(
                200,
                {
                    "model": model_config["display_name"] if model_config else None,
                    "active_model": active_model,
                    "model_loading": loading,
                    "model_error": error,
                    "device": device if model_config else None,
                    "dtype": str(dtype) if model_config else None,
                    "defaults": defaults,
                },
            )
            return

        if path == "/warmup":
            with _RUNTIME_LOCK:
                try:
                    _, device, dtype, _ = get_active_pipeline()
                except NoActiveModelError as exc:
                    self._send_json(409, {"error": str(exc)})
                    return
                self._send_json(
                    200,
                    {"status": "ready", "active_model": _runtime_state_snapshot()[0], "device": device, "dtype": str(dtype)},
                )
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

        if not isinstance(payload, dict):
            self._send_json(400, {"error": "Invalid request body: expected a JSON object"})
            return

        if path == "/models/load":
            try:
                result = load_model(payload.get("model"))
            except UnknownModelError as exc:
                self._send_json(400, {"error": str(exc)})
            except (ModelUnavailableError, RuntimeBusyError) as exc:
                self._send_json(409, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": f"Model load failed: {exc}"})
            else:
                self._send_json(200, result)
            return

        if path == "/upscale_stream":
            return self._handle_upscale_stream(payload)
        if path == "/upscale":
            return self._handle_upscale(payload)
        if path == "/generate_stream":
            return self._handle_generate_stream(payload)
        if path != "/generate":
            self.send_error(404, "Unsupported endpoint")
            return

        with _RUNTIME_LOCK:
            try:
                pipe, device, dtype, model_config = get_active_pipeline()
            except NoActiveModelError as exc:
                self._send_json(409, {"error": str(exc)})
                return

            try:
                params = self._parse_generate_params(payload, model_config["defaults"])
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return

            try:
                generator, seed = build_generator(device, params["seed"])
            except Exception as exc:  # noqa: BLE001
                self._send_json(400, {"error": f"Invalid seed: {exc}"})
                return

            print(
                f"Generating image | prompt='{params['prompt']}' steps={params['steps']} guidance={params['guidance']} size={params['width']}x{params['height']} seed={seed} device={device}"
            )

            try:
                with generation_context(device, dtype):
                    result = pipe(
                        params["prompt"],
                        num_inference_steps=params["steps"],
                        guidance_scale=params["guidance"],
                        height=params["height"],
                        width=params["width"],
                        negative_prompt=params["negative_prompt"] or None,
                        generator=generator,
                        cfg_normalization=model_config["defaults"]["cfg_normalization"],
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
        file_path = OUTPUT_DIR / f"{timestamp}_{params['width']}x{params['height']}_{filename_seed}.png"
        try:
            image.save(file_path, format="PNG")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to save image to disk: {exc}")
            file_path = None

        response = {
            "image": f"data:image/png;base64,{encoded}",
            "meta": {
                "prompt": params["prompt"],
                "steps": params["steps"],
                "guidance": params["guidance"],
                "height": params["height"],
                "width": params["width"],
                "negative_prompt": params["negative_prompt"],
                "seed": seed,
                "device": device,
                "dtype": str(dtype),
                "saved_path": str(file_path) if file_path else None,
            },
        }
        self._send_json(200, response)

    def _handle_upscale(self, payload: dict):
        req_id = _new_request_id()
        try:
            image_b64 = payload.get("image")
            scale = float(payload.get("scale", 2.0))
            out_format = str(payload.get("format", UPSCALE_DEFAULT_FORMAT)).lower()
            return_image = payload.get("return_image", True)
            return_image = False if str(return_image).lower() in {"0", "false", "no"} else bool(return_image)
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": f"Invalid parameter type: {exc}", "request_id": req_id})
            return

        if not image_b64:
            self._send_json(400, {"error": "image is required", "request_id": req_id})
            return

        if scale <= 0:
            self._send_json(400, {"error": "scale must be positive", "request_id": req_id})
            return
        if not math.isfinite(scale):
            self._send_json(400, {"error": "scale must be finite", "request_id": req_id})
            return

        scale = max(1.0, min(scale, MAX_UPSCALE_FACTOR))
        out_fmt = _validate_image_format(out_format)

        try:
            decoded = self._decode_base64_image(image_b64)
            image = Image.open(io.BytesIO(decoded)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": f"Invalid image data: {exc}", "request_id": req_id})
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

        slot_acquired = _wait_for_upscale_slot(UPSCALE_WAIT_TIMEOUT)
        if not slot_acquired:
            self._send_json(429, {"error": "Upscale busy, please retry in a moment", "timeout": UPSCALE_WAIT_TIMEOUT, "request_id": req_id})
            return

        try:
            max_edge = max(target_w, target_h)
            tile, pad = _select_upscale_tile(max_edge)
            with _RUNTIME_LOCK:
                upscaler, device = get_upscaler()
                upscaler.tile = tile
                upscaler.tile_pad = pad
                print(f"[UPSCALE] start req={req_id} size={src_w}x{src_h}->{target_w}x{target_h} scale={scale} fmt={out_fmt} tile={tile} pad={pad}")
                # RealESRGAN expects BGR numpy input; silence verbose tile logs
                img_np = np.array(image)[:, :, ::-1]
                with contextlib.redirect_stdout(io.StringIO()):
                    output, _ = upscaler.enhance(img_np, outscale=scale)
            upscaled = Image.fromarray(output[:, :, ::-1])
            data_url, mime = _encode_image(upscaled, out_fmt)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = OUTPUT_DIR / f"{timestamp}_{target_w}x{target_h}.{out_fmt}"
            _save_image_async(upscaled, file_path, fmt=out_fmt)
            self._send_json(
                200,
                {
                    "image": data_url if return_image else None,
                    "meta": {
                        "type": "upscale",
                        "source_width": src_w,
                        "source_height": src_h,
                        "width": target_w,
                        "height": target_h,
                        "applied_scale": scale,
                        "saved_path": str(file_path),
                        "mime": mime,
                        "request_id": req_id,
                        "tile": tile,
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": f"Upscale failed: {exc}", "request_id": req_id})
        finally:
            _release_upscale_slot(slot_acquired)

    def _handle_upscale_stream(self, payload: dict):
        self._sse_disconnected = False
        req_id = _new_request_id()
        try:
            image_b64 = payload.get("image")
            scale = float(payload.get("scale", 2.0))
            out_format = str(payload.get("format", UPSCALE_DEFAULT_FORMAT)).lower()
            return_image = payload.get("return_image", True)
            return_image = False if str(return_image).lower() in {"0", "false", "no"} else bool(return_image)
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": f"Invalid parameter type: {exc}", "request_id": req_id})
            return

        if not image_b64:
            self._send_json(400, {"error": "image is required", "request_id": req_id})
            return

        if scale <= 0:
            self._send_json(400, {"error": "scale must be positive", "request_id": req_id})
            return
        if not math.isfinite(scale):
            self._send_json(400, {"error": "scale must be finite", "request_id": req_id})
            return

        scale = max(1.0, min(scale, MAX_UPSCALE_FACTOR))

        try:
            decoded = self._decode_base64_image(image_b64)
            image = Image.open(io.BytesIO(decoded)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": f"Invalid image data: {exc}", "request_id": req_id})
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

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def emit_queued(meta):
            return self._send_sse_event("queued", {"waited": meta.get("queued_for_seconds", 0), "timeout": UPSCALE_WAIT_TIMEOUT, "request_id": req_id})

        slot_acquired = _wait_for_upscale_slot(UPSCALE_WAIT_TIMEOUT, emit_queued)
        if not slot_acquired:
            self._send_sse_event("error", {"message": "Upscale busy, please retry shortly", "timeout": UPSCALE_WAIT_TIMEOUT, "request_id": req_id})
            return

        try:
            self._finish_upscale_stream(
                image=image,
                src_w=src_w,
                src_h=src_h,
                target_w=target_w,
                target_h=target_h,
                scale=scale,
                out_format=out_format,
                return_image=return_image,
                req_id=req_id,
            )
        finally:
            _release_upscale_slot(slot_acquired)

    def _finish_upscale_stream(self, *, image, src_w, src_h, target_w, target_h, scale, out_format, return_image, req_id):
        def send_progress(current: int, total: int):
            self._send_sse_event("progress", {"current": current, "total": total, "request_id": req_id})

        class _TileProgressWriter:
            def __init__(self, emitter):
                self.buffer = ""
                self.emit = emitter

            def write(self, data):
                text = str(data).replace("\r", "\n")
                self.buffer += text
                while "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                    self._handle_line(line.strip())
                return len(str(data))

            def flush(self):
                if self.buffer:
                    self._handle_line(self.buffer.strip())
                    self.buffer = ""

            def _handle_line(self, line: str):
                match = re.search(r"Tile\s+(\d+)/(\d+)", line, re.IGNORECASE)
                if match:
                    self.emit(int(match.group(1)), int(match.group(2)))

        tile, pad = _select_upscale_tile(max(target_w, target_h))
        out_fmt = _validate_image_format(out_format)
        try:
            with _RUNTIME_LOCK:
                upscaler, device = get_upscaler()
                upscaler.tile = tile
                upscaler.tile_pad = pad
                print(f"[UPSCALE_STREAM] start req={req_id} size={src_w}x{src_h}->{target_w}x{target_h} scale={scale} fmt={out_fmt} tile={tile} pad={pad}")
                writer = _TileProgressWriter(send_progress)
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    img_np = np.array(image)[:, :, ::-1]
                    output, _ = upscaler.enhance(img_np, outscale=scale)
            upscaled = Image.fromarray(output[:, :, ::-1])
            data_url, mime = _encode_image(upscaled, out_fmt)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = OUTPUT_DIR / f"{timestamp}_{target_w}x{target_h}.{out_fmt}"
            _save_image_async(upscaled, file_path, fmt=out_fmt)
            self._send_sse_event(
                "result",
                {
                    "image": data_url if return_image else None,
                    "meta": {
                        "type": "upscale",
                        "source_width": src_w,
                        "source_height": src_h,
                        "width": target_w,
                        "height": target_h,
                        "applied_scale": scale,
                        "saved_path": str(file_path),
                        "mime": mime,
                        "request_id": req_id,
                        "tile": tile,
                    },
                },
            )
            self._send_sse_event("done", {"ok": True, "request_id": req_id})
        except Exception as exc:  # noqa: BLE001
            self._send_sse_event("error", {"message": f"Upscale failed: {exc}", "request_id": req_id})

    def log_message(self, fmt, *args):  # noqa: D401,N802
        """Silence noisy health polling logs."""
        if getattr(self, "path", "").startswith("/health"):
            return
        return super().log_message(fmt, *args)

    # ==== Streaming generation with progress ====
    def _send_sse_event(self, event: str, data: dict) -> bool:
        if getattr(self, "_sse_disconnected", False):
            return False
        connection = getattr(self, "connection", None)
        previous_timeout = connection.gettimeout() if connection is not None else None
        try:
            if connection is not None:
                connection.settimeout(SSE_WRITE_TIMEOUT)
            message = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            self.wfile.write(message.encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, socket.timeout, TimeoutError):
            self._sse_disconnected = True
            return False
        except Exception as exc:  # noqa: BLE001
            self._sse_disconnected = True
            print(f"SSE send error: {exc}")
            return False
        finally:
            if connection is not None:
                try:
                    connection.settimeout(previous_timeout)
                except OSError:
                    pass

    def _handle_generate_stream(self, payload: dict):
        self._sse_disconnected = False
        with _RUNTIME_LOCK:
            try:
                pipe, device, dtype, model_config = get_active_pipeline()
            except NoActiveModelError as exc:
                self._send_json(409, {"error": str(exc)})
                return

            try:
                params = self._parse_generate_params(payload, model_config["defaults"])
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return

            try:
                generator, seed = build_generator(device, params["seed"])
            except Exception as exc:  # noqa: BLE001
                self._send_json(400, {"error": f"Invalid seed: {exc}"})
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            print("[SSE] start /generate_stream", {'seed': seed, 'size': f"{params['width']}x{params['height']}"})
            print(
                f"[SSE] Generating image | prompt='{params['prompt']}' steps={params['steps']} guidance={params['guidance']} size={params['width']}x{params['height']} seed={seed} device={device}"
            )

            try:
                # Check if pipeline supports callback; fallback to no per-step progress if not.
                pipe_signature = inspect.signature(pipe.__call__)
                supports_callback = "callback" in pipe_signature.parameters
                supports_callback_steps = "callback_steps" in pipe_signature.parameters

                def progress_callback(step: int, _timestep, _latents):
                    # step is zero-based; report human-friendly step count
                    ok = self._send_sse_event(
                        "progress",
                        {"step": step + 1, "total_steps": params["steps"]},
                    )
                    if not ok:
                        return

                with generation_context(device, dtype):
                    kwargs = dict(
                        prompt=params["prompt"],
                        num_inference_steps=params["steps"],
                        guidance_scale=params["guidance"],
                        height=params["height"],
                        width=params["width"],
                        negative_prompt=params["negative_prompt"] or None,
                        generator=generator,
                        cfg_normalization=model_config["defaults"]["cfg_normalization"],
                    )
                    if supports_callback:
                        kwargs["callback"] = progress_callback
                    if supports_callback_steps:
                        kwargs["callback_steps"] = 1
                    else:
                        # Emit a start progress event to indicate fallback mode
                        self._send_sse_event("progress", {"step": 0, "total_steps": params["steps"], "note": "no_callback"})
                    result = pipe(**kwargs)
                image = result.images[0]
            except BrokenPipeError:
                print("[SSE] Client disconnected during generation.")
                return
            except Exception as exc:  # noqa: BLE001
                self._send_sse_event("error", {"error": f"Generation failed: {exc}"})
                return

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_seed = seed if seed is not None else "rand"
        file_path = OUTPUT_DIR / f"{timestamp}_{params['width']}x{params['height']}_{filename_seed}.png"
        try:
            image.save(file_path, format="PNG")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to save image to disk: {exc}")
            file_path = None

        payload = {
            "image": f"data:image/png;base64,{encoded}",
            "meta": {
                "prompt": params["prompt"],
                "steps": params["steps"],
                "guidance": params["guidance"],
                "height": params["height"],
                "width": params["width"],
                "negative_prompt": params["negative_prompt"],
                "seed": seed,
                "device": device,
                "dtype": str(dtype),
                "saved_path": str(file_path) if file_path else None,
            },
        }
        self._send_sse_event("complete", payload)


def run_server():
    if not WEB_DIR.exists():
        print(f"Static directory not found: {WEB_DIR}")
        return

    server = ThreadingHTTPServer((HOST, PORT), WebUIHandler)
    server.daemon_threads = True  # allow Ctrl+C to exit even if requests are running
    print(f"Serving WebUI on http://{HOST}:{PORT}")
    print("No model loaded. Choose a model through the WebUI before generating.")
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
