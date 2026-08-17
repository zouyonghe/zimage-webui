import contextlib
import io
import importlib.util
import json
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from threading import Lock
from unittest.mock import MagicMock, patch


if importlib.util.find_spec("torch") is None:
    torch_stub = types.ModuleType("torch")
    torch_stub.dtype = type("dtype", (), {})
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.float16 = "float16"
    torch_stub.float32 = "float32"
    torch_stub.Generator = MagicMock
    torch_stub.seed = lambda: 1
    torch_stub.no_grad = MagicMock
    torch_stub.autocast = MagicMock
    torch_stub.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        cudnn=types.SimpleNamespace(allow_tf32=False),
    )
    torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        is_bf16_supported=lambda: False,
        empty_cache=lambda: None,
    )
    sys.modules["torch"] = torch_stub

if importlib.util.find_spec("diffusers") is None:
    diffusers_stub = types.ModuleType("diffusers")
    diffusers_stub.ZImagePipeline = type("ZImagePipeline", (), {"from_pretrained": MagicMock()})
    sys.modules["diffusers"] = diffusers_stub

if importlib.util.find_spec("numpy") is None:
    sys.modules["numpy"] = types.ModuleType("numpy")

if importlib.util.find_spec("PIL") is None:
    pil_stub = types.ModuleType("PIL")
    image_stub = types.ModuleType("PIL.Image")
    image_stub.Image = type("Image", (), {})
    image_stub.open = MagicMock()
    pil_stub.Image = image_stub
    sys.modules["PIL"] = pil_stub
    sys.modules["PIL.Image"] = image_stub

import webui_server as server


_DEFAULT_PAYLOAD = object()


class TrackingLock:
    def __init__(self):
        self._lock = Lock()

    def acquire(self, blocking=True, timeout=-1):
        return self._lock.acquire(blocking, timeout)

    def release(self):
        self._lock.release()

    def locked(self):
        return self._lock.locked()

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("failed to acquire tracking lock")
        return self

    def __exit__(self, *_args):
        self.release()


class ModelRuntimeTests(unittest.TestCase):
    def setUp(self):
        server._PIPE = None
        server._DEVICE = None
        server._DTYPE = None
        server._ACTIVE_MODEL_ID = None
        server._MODEL_LOADING = False
        server._PIPE_ERROR = None
        server._RUNTIME_LOCK = Lock()
        server._STATE_LOCK = Lock()

    def _complete_model(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "model_index.json").write_text("{}", encoding="utf-8")
        for component in ("text_encoder", "tokenizer", "transformer", "vae"):
            (root / component).mkdir()
        return root

    def _pipeline(self):
        pipeline = MagicMock(name="pipeline")
        pipeline.to.return_value = pipeline
        return pipeline

    def _handler(self, path: str, payload=_DEFAULT_PAYLOAD):
        handler = object.__new__(server.WebUIHandler)
        raw = json.dumps({} if payload is _DEFAULT_PAYLOAD else payload).encode("utf-8")
        handler.path = path
        handler.headers = {"content-length": str(len(raw))}
        handler.rfile = io.BytesIO(raw)
        handler._send_json = MagicMock()
        return handler

    def test_registry_contains_fixed_model_defaults(self):
        self.assertEqual({"turbo", "base"}, set(server.MODEL_REGISTRY))
        self.assertEqual(server.ROOT / "zimage-model", server.MODEL_REGISTRY["turbo"]["path"])
        self.assertEqual(server.ROOT / "zimage-base-model", server.MODEL_REGISTRY["base"]["path"])
        self.assertEqual(
            {"steps": 9, "guidance": 0.0, "cfg_normalization": False},
            server.MODEL_REGISTRY["turbo"]["defaults"],
        )
        self.assertEqual(
            {"steps": 50, "guidance": 4.0, "cfg_normalization": False},
            server.MODEL_REGISTRY["base"]["defaults"],
        )
        self.assertTrue(server.MODEL_REGISTRY["turbo"]["display_name"])
        self.assertTrue(server.MODEL_REGISTRY["base"]["display_name"])

    def test_startup_has_no_active_model(self):
        status = server.get_models_status()

        self.assertIsNone(status["active_model"])
        self.assertFalse(status["loading"])
        self.assertIsNone(status["last_error"])
        self.assertIsNone(server._PIPE)

    def test_run_server_does_not_preload_pipeline(self):
        http_server = MagicMock()
        with (
            patch.object(server, "ThreadingHTTPServer", return_value=http_server),
            patch.object(server.signal, "signal"),
            patch.object(server.ZImagePipeline, "from_pretrained") as load,
        ):
            server.run_server()

        load.assert_not_called()
        http_server.serve_forever.assert_called_once_with()

    def test_unknown_model_is_rejected_before_runtime_changes(self):
        old_pipeline = self._pipeline()
        server._PIPE = old_pipeline
        server._ACTIVE_MODEL_ID = "turbo"

        with self.assertRaises(server.UnknownModelError):
            server.load_model("unknown")

        self.assertIs(server._PIPE, old_pipeline)
        self.assertEqual("turbo", server._ACTIVE_MODEL_ID)

    def test_incomplete_model_is_rejected_before_unloading_active_pipeline(self):
        old_pipeline = self._pipeline()
        server._PIPE = old_pipeline
        server._ACTIVE_MODEL_ID = "turbo"
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            (model_path / "model_index.json").write_text("{}", encoding="utf-8")
            with patch.dict(server.MODEL_REGISTRY["base"], {"path": model_path}):
                with self.assertRaises(server.ModelUnavailableError):
                    server.load_model("base")

        self.assertIs(server._PIPE, old_pipeline)
        self.assertEqual("turbo", server._ACTIVE_MODEL_ID)

    def test_successful_load_publishes_pipeline_and_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = self._complete_model(Path(temp_dir))
            pipeline = self._pipeline()
            with (
                patch.dict(server.MODEL_REGISTRY["base"], {"path": model_path}),
                patch.object(server.torch.cuda, "is_available", return_value=True),
                patch.object(server.torch.cuda, "is_bf16_supported", return_value=True),
                patch.object(server.ZImagePipeline, "from_pretrained", return_value=pipeline) as load,
            ):
                result = server.load_model("base")

        load.assert_called_once_with(str(model_path), torch_dtype=server.torch.bfloat16, local_files_only=True)
        self.assertIs(server._PIPE, pipeline)
        self.assertEqual("base", server._ACTIVE_MODEL_ID)
        self.assertEqual(server.MODEL_REGISTRY["base"]["defaults"], result["defaults"])
        self.assertIsNone(server._PIPE_ERROR)

    def test_switching_releases_old_pipeline_before_loading_new_one(self):
        old_pipeline = self._pipeline()
        new_pipeline = self._pipeline()
        server._PIPE = old_pipeline
        server._DEVICE = "cuda"
        server._DTYPE = server.torch.float16
        server._ACTIVE_MODEL_ID = "turbo"
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = self._complete_model(Path(temp_dir))

            def load_new_pipeline(*_args, **_kwargs):
                self.assertIsNone(server._PIPE)
                self.assertIsNone(server._ACTIVE_MODEL_ID)
                return new_pipeline

            with (
                patch.dict(server.MODEL_REGISTRY["base"], {"path": model_path}),
                patch.object(server.torch.cuda, "is_available", return_value=True),
                patch.object(server.torch.cuda, "is_bf16_supported", return_value=False),
                patch.object(server.torch.cuda, "empty_cache") as empty_cache,
                patch.object(server.gc, "collect") as collect,
                patch.object(server.ZImagePipeline, "from_pretrained", side_effect=load_new_pipeline),
            ):
                server.load_model("base")

        collect.assert_called()
        empty_cache.assert_called()
        self.assertIs(server._PIPE, new_pipeline)
        self.assertEqual("base", server._ACTIVE_MODEL_ID)

    def test_failed_load_leaves_no_active_model_and_exposes_error(self):
        server._PIPE = self._pipeline()
        server._ACTIVE_MODEL_ID = "turbo"
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = self._complete_model(Path(temp_dir))
            with (
                patch.dict(server.MODEL_REGISTRY["base"], {"path": model_path}),
                patch.object(server.torch.cuda, "is_available", return_value=True),
                patch.object(server.torch.cuda, "is_bf16_supported", return_value=True),
                patch.object(server.ZImagePipeline, "from_pretrained", side_effect=RuntimeError("load exploded")),
            ):
                with self.assertRaisesRegex(RuntimeError, "load exploded"):
                    server.load_model("base")

        self.assertIsNone(server._PIPE)
        self.assertIsNone(server._ACTIVE_MODEL_ID)
        self.assertEqual("load exploded", server._PIPE_ERROR)
        self.assertFalse(server._MODEL_LOADING)

    def test_get_active_pipeline_rejects_no_model(self):
        with self.assertRaises(server.NoActiveModelError):
            server.get_active_pipeline()

    def test_generate_endpoint_returns_409_without_active_model(self):
        handler = self._handler("/generate")

        handler.do_POST()

        status, payload = handler._send_json.call_args.args
        self.assertEqual(409, status)
        self.assertIn("No model", payload["error"])

    def test_stream_endpoint_returns_409_without_active_model(self):
        handler = self._handler("/generate_stream")

        handler.do_POST()

        self.assertEqual(409, handler._send_json.call_args.args[0])

    def test_models_endpoint_returns_runtime_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            turbo_path = self._complete_model(root / "turbo")
            base_path = root / "base"
            base_path.mkdir()
            server._ACTIVE_MODEL_ID = "turbo"
            server._MODEL_LOADING = True
            server._PIPE_ERROR = "last failure"
            handler = self._handler("/models")
            with (
                patch.dict(server.MODEL_REGISTRY["turbo"], {"path": turbo_path}),
                patch.dict(server.MODEL_REGISTRY["base"], {"path": base_path}),
            ):
                handler.do_GET()

        status, payload = handler._send_json.call_args.args
        self.assertEqual(200, status)
        self.assertEqual({"models", "active_model", "loading", "last_error"}, set(payload))
        self.assertEqual("turbo", payload["active_model"])
        self.assertTrue(payload["loading"])
        self.assertEqual("last failure", payload["last_error"])
        entries = {entry["id"]: entry for entry in payload["models"]}
        self.assertEqual(
            {"id", "display_name", "path", "available", "missing", "defaults"},
            set(entries["turbo"]),
        )
        self.assertTrue(entries["turbo"]["available"])
        self.assertFalse(entries["base"]["available"])
        self.assertEqual(server.MODEL_REGISTRY["base"]["defaults"], entries["base"]["defaults"])

    def test_models_status_uses_one_coherent_state_snapshot(self):
        snapshot = ("base", True, "loading base", "cuda", server.torch.float16)

        def mutate_live_globals(_path):
            server._ACTIVE_MODEL_ID = "turbo"
            server._MODEL_LOADING = False
            server._PIPE_ERROR = None
            return []

        with (
            patch.object(server, "_runtime_state_snapshot", return_value=snapshot) as read_state,
            patch.object(server, "_missing_model_paths", side_effect=mutate_live_globals),
        ):
            status = server.get_models_status()

        read_state.assert_called_once_with()
        self.assertEqual("base", status["active_model"])
        self.assertTrue(status["loading"])
        self.assertEqual("loading base", status["last_error"])

    def test_load_publishes_inactive_loading_snapshot_before_pipeline_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = self._complete_model(Path(temp_dir))
            pipeline = self._pipeline()

            def inspect_loading_state(*_args, **_kwargs):
                status = server.get_models_status()
                self.assertIsNone(status["active_model"])
                self.assertTrue(status["loading"])
                self.assertIsNone(status["last_error"])
                return pipeline

            with (
                patch.dict(server.MODEL_REGISTRY["base"], {"path": model_path}),
                patch.object(server.torch.cuda, "is_available", return_value=True),
                patch.object(server.torch.cuda, "is_bf16_supported", return_value=True),
                patch.object(server.ZImagePipeline, "from_pretrained", side_effect=inspect_loading_state),
            ):
                server.load_model("base")

    def test_info_uses_coherent_state_snapshot(self):
        handler = self._handler("/info")
        snapshot = ("base", True, "loading base", "cuda", server.torch.float16)
        with patch.object(server, "_runtime_state_snapshot", return_value=snapshot) as read_state:
            handler.do_GET()

        read_state.assert_called_once_with()
        payload = handler._send_json.call_args.args[1]
        self.assertEqual(("base", True, "loading base", "cuda"), (
            payload["active_model"], payload["model_loading"], payload["model_error"], payload["device"]
        ))

    def test_health_uses_coherent_state_snapshot(self):
        handler = self._handler("/health")
        snapshot = (None, True, "loading base", None, None)
        with patch.object(server, "_runtime_state_snapshot", return_value=snapshot) as read_state:
            handler.do_GET()

        read_state.assert_called_once_with()
        payload = handler._send_json.call_args.args[1]
        self.assertFalse(payload["pipeline_loaded"])
        self.assertIsNone(payload["active_model"])
        self.assertTrue(payload["model_loading"])
        self.assertEqual("loading base", payload["model_error"])

    def test_model_load_endpoint_maps_errors_to_status_codes(self):
        cases = (
            (server.UnknownModelError("unknown"), 400),
            (server.ModelUnavailableError("incomplete"), 409),
            (server.RuntimeBusyError("busy"), 409),
            (RuntimeError("failed"), 500),
        )
        for error, expected_status in cases:
            with self.subTest(error=type(error).__name__):
                handler = self._handler("/models/load", {"model": "base"})
                with patch.object(server, "load_model", side_effect=error):
                    handler.do_POST()
                self.assertEqual(expected_status, handler._send_json.call_args.args[0])

    def test_model_load_endpoint_returns_active_defaults(self):
        handler = self._handler("/models/load", {"model": "base"})
        result = {"active_model": "base", "defaults": server.MODEL_REGISTRY["base"]["defaults"]}
        with patch.object(server, "load_model", return_value=result):
            handler.do_POST()

        self.assertEqual((200, result), handler._send_json.call_args.args)

    def test_warmup_does_not_choose_model(self):
        handler = self._handler("/warmup")

        handler.do_GET()

        self.assertEqual(409, handler._send_json.call_args.args[0])
        self.assertIsNone(server._ACTIVE_MODEL_ID)

    def test_warmup_reports_ready_active_model(self):
        server._PIPE = self._pipeline()
        server._DEVICE = "cuda"
        server._DTYPE = server.torch.float16
        server._ACTIVE_MODEL_ID = "base"
        handler = self._handler("/warmup")

        handler.do_GET()

        status, payload = handler._send_json.call_args.args
        self.assertEqual(200, status)
        self.assertEqual("ready", payload["status"])
        self.assertEqual("base", payload["active_model"])

    def test_load_model_fails_fast_while_runtime_is_busy(self):
        server._RUNTIME_LOCK.acquire()
        try:
            with (
                patch.object(server, "_missing_model_paths") as validate,
                self.assertRaises(server.RuntimeBusyError),
            ):
                server.load_model("base")
        finally:
            server._RUNTIME_LOCK.release()
        validate.assert_not_called()

    def test_non_object_json_body_returns_400(self):
        for payload in (None, [], "base", 1):
            with self.subTest(payload=payload):
                handler = self._handler("/models/load", payload)
                handler.do_POST()
                status, response = handler._send_json.call_args.args
                self.assertEqual(400, status)
                self.assertIn("JSON object", response["error"])

    def test_non_string_model_id_returns_400(self):
        handler = self._handler("/models/load", {"model": []})

        handler.do_POST()

        self.assertEqual(400, handler._send_json.call_args.args[0])

    def test_health_and_info_report_inactive_state(self):
        for path in ("/health", "/info"):
            with self.subTest(path=path):
                handler = self._handler(path)
                handler.do_GET()
                status, payload = handler._send_json.call_args.args
                self.assertEqual(200, status)
                self.assertIsNone(payload["active_model"])
                self.assertFalse(payload["model_loading"])
                self.assertIsNone(payload["model_error"])
                if path == "/info":
                    self.assertIsNone(payload["model"])
                    self.assertIsNone(payload["device"])
                    self.assertIsNone(payload["dtype"])
                    self.assertIsNone(payload["defaults"])

    def test_health_and_info_report_active_model_defaults(self):
        server._PIPE = self._pipeline()
        server._DEVICE = "cuda"
        server._DTYPE = server.torch.float16
        server._ACTIVE_MODEL_ID = "base"
        server._PIPE_ERROR = "previous error"

        health = self._handler("/health")
        health.do_GET()
        health_payload = health._send_json.call_args.args[1]
        self.assertTrue(health_payload["pipeline_ready"])
        self.assertEqual("base", health_payload["active_model"])
        self.assertEqual("previous error", health_payload["model_error"])

        info = self._handler("/info")
        info.do_GET()
        info_payload = info._send_json.call_args.args[1]
        self.assertEqual("base", info_payload["active_model"])
        self.assertEqual("Z-Image", info_payload["model"])
        self.assertEqual(50, info_payload["defaults"]["steps"])
        self.assertEqual(4.0, info_payload["defaults"]["guidance"])
        self.assertFalse(info_payload["defaults"]["cfg_normalization"])

    def test_regular_generation_passes_active_cfg_normalization(self):
        pipeline = MagicMock()
        pipeline.return_value.images = [MagicMock()]
        server._PIPE = pipeline
        server._DEVICE = "cuda"
        server._DTYPE = server.torch.float16
        server._ACTIVE_MODEL_ID = "base"
        handler = self._handler("/generate")
        server._RUNTIME_LOCK = TrackingLock()

        def generate(*_args, **_kwargs):
            self.assertTrue(server._RUNTIME_LOCK.locked())
            return types.SimpleNamespace(images=[MagicMock()])

        pipeline.side_effect = generate

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(server, "OUTPUT_DIR", Path(temp_dir) / "outputs"),
            patch.object(server, "build_generator", return_value=(MagicMock(), 7)),
            patch.object(server, "generation_context", return_value=contextlib.nullcontext()),
            patch.dict(server.MODEL_REGISTRY["base"]["defaults"], {"cfg_normalization": True}),
        ):
            handler.do_POST()

        self.assertEqual(50, pipeline.call_args.kwargs["num_inference_steps"])
        self.assertEqual(4.0, pipeline.call_args.kwargs["guidance_scale"])
        self.assertTrue(pipeline.call_args.kwargs["cfg_normalization"])
        self.assertEqual(200, handler._send_json.call_args.args[0])

    def test_explicit_generation_values_override_active_defaults(self):
        params = server.WebUIHandler._parse_generate_params(
            MagicMock(),
            {"steps": 7, "guidance": 1.5},
            server.MODEL_REGISTRY["base"]["defaults"],
        )

        self.assertEqual(7, params["steps"])
        self.assertEqual(1.5, params["guidance"])

    def test_generation_rejects_non_finite_guidance(self):
        with self.assertRaisesRegex(ValueError, "guidance must be finite"):
            server.WebUIHandler._parse_generate_params(
                MagicMock(),
                {"guidance": "nan"},
                server.MODEL_REGISTRY["base"]["defaults"],
            )

    def test_stream_generation_passes_active_cfg_normalization(self):
        class RecordingPipeline:
            def __init__(self):
                self.kwargs = None

            def __call__(self, **kwargs):
                self_test.assertTrue(server._RUNTIME_LOCK.locked())
                self.kwargs = kwargs
                return types.SimpleNamespace(images=[MagicMock()])

        self_test = self
        pipeline = RecordingPipeline()
        server._PIPE = pipeline
        server._DEVICE = "cuda"
        server._DTYPE = server.torch.float16
        server._ACTIVE_MODEL_ID = "base"
        handler = self._handler("/generate_stream")
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler._send_sse_event = MagicMock(return_value=True)
        server._RUNTIME_LOCK = TrackingLock()

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(server, "OUTPUT_DIR", Path(temp_dir) / "outputs"),
            patch.object(server, "build_generator", return_value=(MagicMock(), 7)),
            patch.object(server, "generation_context", return_value=contextlib.nullcontext()),
        ):
            handler.do_POST()

        self.assertFalse(pipeline.kwargs["cfg_normalization"])
        handler._send_sse_event.assert_any_call("complete", unittest.mock.ANY)

    def _assert_upscale_holds_runtime_lock(self, streaming: bool):
        handler = self._handler("/upscale_stream" if streaming else "/upscale")
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler._send_sse_event = MagicMock(return_value=True)
        fake_image = MagicMock(size=(16, 16))
        fake_image.convert.return_value = fake_image
        fake_upscaler = MagicMock()
        server._RUNTIME_LOCK = TrackingLock()

        def enhance(*_args, **_kwargs):
            self.assertTrue(server._RUNTIME_LOCK.locked())
            return MagicMock(), None

        fake_upscaler.enhance.side_effect = enhance
        with (
            patch.object(handler, "_decode_base64_image", return_value=b"image"),
            patch.object(server.Image, "open", return_value=fake_image),
            patch.object(server.Image, "fromarray", return_value=fake_image, create=True),
            patch.object(server.np, "array", return_value=MagicMock(), create=True),
            patch.object(server, "_wait_for_upscale_slot", return_value=True),
            patch.object(server, "_release_upscale_slot"),
            patch.object(server, "get_upscaler", return_value=(fake_upscaler, "cuda")),
            patch.object(server, "_encode_image", return_value=("data:image/png;base64,x", "image/png")),
            patch.object(server, "_save_image_async"),
        ):
            if streaming:
                handler._handle_upscale_stream({"image": "x"})
            else:
                handler._handle_upscale({"image": "x"})

        fake_upscaler.enhance.assert_called_once()

    def test_regular_upscale_holds_runtime_lock(self):
        self._assert_upscale_holds_runtime_lock(streaming=False)

    def test_stream_upscale_holds_runtime_lock(self):
        self._assert_upscale_holds_runtime_lock(streaming=True)

    def _assert_upscale_releases_slot_after_failure(self, *, streaming: bool, fail_at: str):
        handler = self._handler("/upscale_stream" if streaming else "/upscale")
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler._send_sse_event = MagicMock(return_value=True)
        fake_image = MagicMock(size=(16, 16))
        fake_image.convert.return_value = fake_image
        fake_upscaler = MagicMock()
        fake_upscaler.enhance.return_value = (MagicMock(), None)
        semaphore = MagicMock()
        semaphore.acquire.return_value = True
        patches = [
            patch.object(handler, "_decode_base64_image", return_value=b"image"),
            patch.object(server.Image, "open", return_value=fake_image),
            patch.object(server.Image, "fromarray", return_value=fake_image, create=True),
            patch.object(server.np, "array", return_value=MagicMock(), create=True),
            patch.object(server, "get_upscaler", return_value=(fake_upscaler, "cuda")),
            patch.object(server, "_UPSCALE_SEMAPHORE", semaphore),
            patch.object(server, "_encode_image", return_value=("data:image/png;base64,x", "image/png")),
            patch.object(server, "_save_image_async"),
        ]
        if fail_at == "encode":
            patches[-2] = patch.object(server, "_encode_image", side_effect=RuntimeError("encode failed"))
        else:
            patches[-1] = patch.object(server, "_save_image_async", side_effect=RuntimeError("save failed"))

        with contextlib.ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            if streaming:
                handler._handle_upscale_stream({"image": "x"})
            else:
                handler._handle_upscale({"image": "x"})

        semaphore.acquire.assert_called_once_with(blocking=False)
        semaphore.release.assert_called_once_with()

    def test_regular_upscale_releases_slot_when_encoding_fails(self):
        self._assert_upscale_releases_slot_after_failure(streaming=False, fail_at="encode")

    def test_stream_upscale_releases_slot_when_async_save_fails(self):
        self._assert_upscale_releases_slot_after_failure(streaming=True, fail_at="save")

    def test_sse_write_timeout_marks_client_disconnected(self):
        handler = self._handler("/generate_stream")
        handler.wfile = MagicMock()
        handler.connection = MagicMock()
        handler.connection.gettimeout.return_value = 30.0
        handler.wfile.write.side_effect = socket.timeout("slow client")

        self.assertFalse(handler._send_sse_event("progress", {"step": 1}))
        self.assertFalse(handler._send_sse_event("progress", {"step": 2}))

        self.assertTrue(handler._sse_disconnected)
        handler.wfile.write.assert_called_once()
        self.assertEqual(
            [unittest.mock.call(server.SSE_WRITE_TIMEOUT), unittest.mock.call(30.0)],
            handler.connection.settimeout.call_args_list,
        )

    def test_sse_progress_disconnect_does_not_abort_inference(self):
        class CallbackPipeline:
            def __init__(self):
                self.completed = False

            def __call__(self, callback=None, callback_steps=None, **_kwargs):
                callback(0, None, None)
                self.completed = True
                return types.SimpleNamespace(images=[MagicMock()])

        pipeline = CallbackPipeline()
        server._PIPE = pipeline
        server._DEVICE = "cuda"
        server._DTYPE = server.torch.float16
        server._ACTIVE_MODEL_ID = "base"
        handler = self._handler("/generate_stream")
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler._send_sse_event = MagicMock(return_value=False)
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(server, "OUTPUT_DIR", Path(temp_dir) / "outputs"),
            patch.object(server, "build_generator", return_value=(MagicMock(), 7)),
            patch.object(server, "generation_context", return_value=contextlib.nullcontext()),
        ):
            handler.do_POST()

        self.assertTrue(pipeline.completed)


if __name__ == "__main__":
    unittest.main()
