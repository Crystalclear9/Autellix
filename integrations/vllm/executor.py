"""KV transfer and metadata handling for the pinned tensor-parallel workers."""
from vllm.distributed import broadcast_tensor_dict
from vllm.executor.multiproc_gpu_executor import MultiprocessingGPUExecutor
from vllm.worker.worker import Worker
from vllm.worker.worker_base import WorkerInput, extract_previous_hidden_states

from autellix.runtime.swap import attach_cache_engine


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
