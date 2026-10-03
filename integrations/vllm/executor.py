"""Pinned tensor-parallel workers: install KV transfer hooks on every rank."""
from vllm.executor.multiproc_gpu_executor import MultiprocessingGPUExecutor
from vllm.worker.worker import Worker

from autellix.runtime.swap import attach_cache_engine


class AutellixWorker(Worker):
    def initialize_cache(self, *args, **kwargs):
        result = super().initialize_cache(*args, **kwargs)
        for cache in self.cache_engine:
            attach_cache_engine(cache)
        return result


class AutellixMPExecutor(MultiprocessingGPUExecutor):
    def _get_worker_module_and_class(self):
        return "integrations.vllm.executor", "AutellixWorker", None
