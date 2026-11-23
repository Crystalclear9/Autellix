"""KV transfer and metadata handling for the pinned vLLM workers."""
from vllm.distributed import broadcast_tensor_dict
from vllm.executor.gpu_executor import GPUExecutor
from vllm.executor.multiproc_gpu_executor import MultiprocessingGPUExecutor
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.worker import Worker
from vllm.worker.worker_base import WorkerInput, extract_previous_hidden_states

from autellix.runtime.swap import attach_cache_engine, packed_host_swap


class BlockMajorHostCacheEngine(CacheEngine):
    def _allocate_kv_cache(self, num_blocks, device):
        if device != "cpu":
            return super()._allocate_kv_cache(num_blocks, device)
        import torch
        from autellix.runtime.packed_kv import PackedKVLayout

        self.packed_gpu = PackedKVLayout(self.gpu_cache, channels=2, token_major=True)
        self.host_blocks = torch.zeros(self.packed_gpu.shape(num_blocks), dtype=self.dtype,
                                       device="cpu", pin_memory=True)
        # The logical layer views share the same block-major host storage.
        return [self.host_blocks[:, 2*i:2*i+2].transpose(0, 1) for i in range(self.num_attention_layers)]

    def swap_in(self, mapping):
        return packed_host_swap(self.packed_gpu, self.host_blocks, mapping, to_host=False)

    def swap_out(self, mapping):
        return packed_host_swap(self.packed_gpu, self.host_blocks, mapping, to_host=True)


class AutellixPackedHostWorker(Worker):
    def _init_cache_engine(self):
        cache = BlockMajorHostCacheEngine(self.cache_config, self.model_config,
                                         self.parallel_config, self.device_config)
        self.cache_engine = [cache]
        self.gpu_cache = [cache.gpu_cache]


class AutellixPackedHostGPUExecutor(GPUExecutor):
    def _get_worker_module_and_class(self):
        return __name__, AutellixPackedHostWorker.__name__, None


class AutellixWorker(Worker):
    batched_swap = True
    fused_swap = False

    def initialize_cache(self, *args, **kwargs):
        result = super().initialize_cache(*args, **kwargs)
        if self.batched_swap:
            for cache in self.cache_engine:
                attach_cache_engine(cache, fused=self.fused_swap)
        return result

    def _get_worker_input_from_broadcast(self):
        data = broadcast_tensor_dict(src=0)
        if not data:
            return None
        worker_input = WorkerInput.from_broadcasted_tensor_dict(data)
        if worker_input.num_seq_groups == 0:
            # Transfer-only steps carry worker inputs but no attention metadata.
            # Worker.execute_model runs the cache operations and returns [].
            return None, worker_input, {}
        model_input = self.model_runner.make_model_input_from_broadcasted_tensor_dict(data)
        return model_input, worker_input, extract_previous_hidden_states(data)


class AutellixNativeSwapWorker(AutellixWorker):
    batched_swap = False


class AutellixFusedSwapWorker(AutellixWorker):
    fused_swap = True


class AutellixMPExecutor(MultiprocessingGPUExecutor):
    worker_class = AutellixWorker

    def _get_worker_module_and_class(self):
        return __name__, self.worker_class.__name__, None


class AutellixNativeSwapMPExecutor(AutellixMPExecutor):
    worker_class = AutellixNativeSwapWorker


class AutellixFusedSwapMPExecutor(AutellixMPExecutor):
    worker_class = AutellixFusedSwapWorker
