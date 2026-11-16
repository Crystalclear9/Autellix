import sys
import unittest
from types import ModuleType
from unittest.mock import patch, Mock

from integrations.sglang.backend import SGLangBackend
from integrations.sglang.legacy_gpu import run_native_scheduler_process


class NativeBaselineTests(unittest.TestCase):
    def test_native_target_only_installs_hardware_compatibility(self):
        module = ModuleType("sglang.srt.managers.scheduler")
        module.Scheduler = type("Scheduler", (), {})
        module.run_scheduler_process = Mock(return_value="native-result")
        with patch.dict(sys.modules, {module.__name__: module}), \
             patch("integrations.sglang.legacy_gpu.install_native_ops") as kernels, \
             patch("integrations.sglang.scheduler.install") as hooks:
            self.assertEqual(run_native_scheduler_process(1, token=2, attention_score_elements=2**24),
                             "native-result")
        kernels.assert_called_once_with(prefix_attention=True, attention_score_elements=2**24)
        hooks.assert_not_called()
        module.run_scheduler_process.assert_called_once_with(1, token=2)

    def test_native_rejects_extra_autellix_reserves(self):
        with patch("integrations.sglang.backend.version", return_value="0.4.9.post6"):
            with self.assertRaisesRegex(ValueError, "overprovision=0"):
                SGLangBackend("unused", "unused", autellix_native_baseline=True)


if __name__ == "__main__":
    unittest.main()
