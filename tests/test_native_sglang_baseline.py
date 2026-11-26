import sys
import unittest
from types import ModuleType
from unittest.mock import patch, Mock

from integrations.sglang.backend import SGLangBackend
from integrations.sglang.legacy_gpu import run_native_scheduler_process, token_chunked_forward


class NativeBaselineTests(unittest.TestCase):
    def test_dense_mlp_chunks_preserve_values_and_bound_intermediates(self):
        import torch
        generator = torch.Generator().manual_seed(0)
        weights = torch.randn(7, 22, generator=generator)
        down = torch.randn(11, 7, generator=generator)
        calls = []
        def forward(self, x):
            calls.append(x.shape[0])
            gate, up = (x @ weights).chunk(2, dim=-1)
            return (torch.nn.functional.silu(gate) * up) @ down
        x = torch.randn(1031, 7, generator=generator)
        expected = forward(None, x)
        calls.clear()
        with torch.inference_mode():
            result = token_chunked_forward(forward, 128)(None, x)
        torch.testing.assert_close(result, expected)
        self.assertEqual(sum(calls), len(x))
        self.assertLessEqual(max(calls), 128)

    def test_native_target_only_installs_hardware_compatibility(self):
        module = ModuleType("sglang.srt.managers.scheduler")
        module.Scheduler = type("Scheduler", (), {})
        module.run_scheduler_process = Mock(return_value="native-result")
        with patch.dict(sys.modules, {module.__name__: module}), \
             patch("integrations.sglang.legacy_gpu.install_native_ops") as kernels, \
             patch("integrations.sglang.scheduler.install") as hooks:
            self.assertEqual(run_native_scheduler_process(1, token=2, attention_score_elements=2**24),
                             "native-result")
        kernels.assert_called_once_with(prefix_attention=True, attention_score_elements=2**24, mlp_chunk_size=0)
        hooks.assert_not_called()
        module.run_scheduler_process.assert_called_once_with(1, token=2)

    def test_native_rejects_extra_autellix_reserves(self):
        with patch("integrations.sglang.backend.version", return_value="0.4.9.post6"):
            with self.assertRaisesRegex(ValueError, "overprovision=0"):
                SGLangBackend("unused", "unused", autellix_native_baseline=True)

    def test_lazy_resolve_requires_stable_decode_overlap(self):
        with patch("integrations.sglang.backend.version", return_value="0.4.9.post6"):
            with self.assertRaisesRegex(ValueError, "requires autellix_stable_decode_overlap"):
                SGLangBackend("unused", "unused", autellix_overlap_lazy_resolve=True)

    def test_modern_pipeline_native_target_loads_tied_embedding_compatibility(self):
        module = ModuleType("sglang.srt.managers.scheduler")
        module.Scheduler = type("Scheduler", (), {})
        module.run_scheduler_process = Mock(return_value="native-result")
        with patch.dict(sys.modules, {module.__name__: module}), \
             patch("integrations.sglang.legacy_gpu.install_native_ops") as kernels, \
             patch("integrations.sglang.pipeline.install_pipeline_embeddings") as embeddings:
            self.assertEqual(run_native_scheduler_process(1, token=2, legacy_gpu=False, pipeline=True),
                             "native-result")
        kernels.assert_not_called()
        embeddings.assert_called_once_with()
        module.run_scheduler_process.assert_called_once_with(1, token=2)


if __name__ == "__main__":
    unittest.main()
