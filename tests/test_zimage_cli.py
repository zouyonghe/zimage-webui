import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_zimage_module():
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        cudnn=types.SimpleNamespace(allow_tf32=False),
    )
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
    )

    diffusers = types.ModuleType("diffusers")
    diffusers.ZImagePipeline = type("ZImagePipeline", (), {})

    spec = importlib.util.spec_from_file_location("zimage_cli_test_target", ROOT / "zimage.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": torch, "diffusers": diffusers}):
        spec.loader.exec_module(module)
    return module


class ZImageCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zimage = load_zimage_module()

    def test_model_directories_are_relative_to_script_root(self):
        self.assertEqual(ROOT, self.zimage.ROOT)
        self.assertEqual(ROOT / "zimage-model", self.zimage.MODEL_CONFIGS["turbo"]["path"])
        self.assertEqual(ROOT / "zimage-base-model", self.zimage.MODEL_CONFIGS["base"]["path"])

    def test_parser_defaults_to_turbo_and_accepts_only_known_models(self):
        parser = self.zimage.build_parser()

        self.assertEqual("turbo", parser.parse_args([]).model)
        for model_id in ("turbo", "base"):
            self.assertEqual(model_id, parser.parse_args(["--model", model_id]).model)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--model", "unknown"])

    def _assert_model_defaults(self, model_id, steps, guidance):
        image = MagicMock()
        pipeline = MagicMock()
        pipeline.return_value = types.SimpleNamespace(images=[image])

        with tempfile.TemporaryDirectory() as temp_dir:
            old_cwd = Path.cwd()
            os.chdir(temp_dir)
            try:
                with (
                    patch.object(sys, "argv", ["zimage.py", "--model", model_id]),
                    patch.object(self.zimage, "load_pipeline", return_value=pipeline) as load,
                ):
                    self.zimage.main()
            finally:
                os.chdir(old_cwd)

        load.assert_called_once_with(model_id, self.zimage.MODEL_CONFIGS[model_id]["path"])
        self.assertEqual(steps, pipeline.call_args.kwargs["num_inference_steps"])
        self.assertEqual(guidance, pipeline.call_args.kwargs["guidance_scale"])
        self.assertFalse(pipeline.call_args.kwargs["cfg_normalization"])
        image.save.assert_called_once_with(Path("zimage_test.png"))

    def test_turbo_generation_defaults(self):
        self._assert_model_defaults("turbo", steps=9, guidance=0.0)

    def test_base_generation_defaults(self):
        self._assert_model_defaults("base", steps=50, guidance=4.0)


if __name__ == "__main__":
    unittest.main()
