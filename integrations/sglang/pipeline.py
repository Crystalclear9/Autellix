"""Replicate stage-zero policy inputs through SGLang's pipeline control channel.

Each stage owns its layer-local KV cache. Only stage zero updates the shared
program table. Database reads and policy times travel with each microbatch so
all stages perform the same admission, preemption and restoration decisions.
"""
import copy
import json
import time


def install_pipeline_embeddings():
    """Load tied Llama output embeddings on the pipeline's last stage."""
    from sglang.srt.models.llama import LlamaForCausalLM, ParallelLMHead
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    original_init = LlamaForCausalLM.__init__
    original_load = LlamaForCausalLM.load_weights

    def init(self, config, quant_config=None, prefix=""):
        original_init(self, config, quant_config, prefix)
        self._autellix_tied_pp_head = (self.pp_group.world_size > 1
                                      and self.pp_group.is_last_rank and config.tie_word_embeddings)
        if self._autellix_tied_pp_head:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size,
                                          quant_config=quant_config, prefix="lm_head")

    def load(self, weights):
        def with_tied_head():
            for name, weight in weights:
                if self._autellix_tied_pp_head and name == "model.embed_tokens.weight":
                    parameter = self.lm_head.weight
                    getattr(parameter, "weight_loader", default_weight_loader)(parameter, weight)
                yield name, weight
        return original_load(self, with_tied_head())

    LlamaForCausalLM.__init__, LlamaForCausalLM.load_weights = init, load


class PolicyTape:
    def __init__(self, leader):
        self.leader = leader
        self.entries = []
        self.position = 0

    def reset(self, entries=None):
        if not self.leader and self.position != len(self.entries):
            raise RuntimeError("pipeline policy inputs were not fully consumed")
        self.entries = [] if entries is None else entries
        self.position = 0

    def value(self, name, function):
        if self.leader:
            result = function()
            self.entries.append((name, copy.deepcopy(result)))
            return result
        expected, result = self.entries[self.position]
        self.position += 1
        if expected != name:
            raise RuntimeError(f"pipeline policy diverged: expected {expected}, received {name}")
        return copy.deepcopy(result)

    def clock(self):
        return self.value("clock", time.monotonic)


class PipelineTable:
    def __init__(self, tape, table):
        self.tape, self.table = tape, table

    def __getattr__(self, name):
        def invoke(*args, **kwargs):
            return self.tape.value("table." + name,
                                   lambda: getattr(self.table, name)(*args, **kwargs))
        return invoke


def configure(scheduler, options):
    from autellix.runtime.policy import PolicyConfig, ProgramTable, RuntimeScheduler
    from autellix.runtime.window import SchedulingWindow
    import sglang.srt.managers.scheduler as module

    leader = scheduler.pp_rank == 0
    tape = scheduler._autellix_tape = PolicyTape(leader)
    table = ProgramTable(options["table_path"]) if leader else None
    controller = RuntimeScheduler(PipelineTable(tape, table), PolicyConfig(**options["config"]),
                                  options.get("trace") if leader else None, clock=tape.clock,
                                  publish_activity_at_window=options.get("publish_activity_at_window", False),
                                  publish_activity=options.get("publish_activity", True))
    if not leader and options.get("trace"):
        def emit(event, **fields):
            timestamp = tape.clock()
            if event in {"batch", "execute", "finished", "cancelled", "swap_out", "swap_in"}:
                with open(f'{options["trace"]}.stage-{scheduler.pp_rank}', "a") as stream:
                    stream.write(json.dumps(dict(event=event, time=timestamp, **fields)) + "\n")
        controller.emit = emit
    scheduler.autellix = controller
    capacity = options["capacity"] // scheduler.pp_size
    scheduler._autellix_windows = [
        SchedulingWindow(controller, capacity,
                         refill_idle_slots=options.get("refill_idle_slots", False),
                         resident_reserve_only=options.get("resident_reserve_only", False))
        for _ in range(scheduler.pp_size)
    ]
    scheduler._autellix_reserves = [{} for _ in range(scheduler.pp_size)]
    scheduler._autellix_microbatch = -1
    scheduler._autellix_inflight = set()
    scheduler._autellix_batches = [None] * scheduler.pp_size
    original_recv = scheduler.recv_requests

    def recv():
        requests = original_recv()
        if leader:
            tape.reset()
        else:
            requests, entries = requests
            tape.reset(entries)
        scheduler._autellix_microbatch = (scheduler._autellix_microbatch + 1) % scheduler.pp_size
        mb = scheduler._autellix_microbatch
        scheduler._autellix_window = scheduler._autellix_windows[mb]
        scheduler._autellix_reserve = scheduler._autellix_reserves[mb]
        return requests

    scheduler.recv_requests = recv
    original_transfer = module.point_to_point_pyobj

    def transfer(data, rank, group, src, dst):
        if rank == src:
            data = (data, tape.entries)
        return original_transfer(data, rank, group, src, dst)

    module.point_to_point_pyobj = transfer


def abort_inflight(scheduler, request):
    # Newly prefilling microbatches are not yet in running_mbs. Keep the actual
    # forward batches so cancellation reaches those requests as well as decodes.
    for batch in scheduler._autellix_batches:
        if batch is not None:
            for req in batch.reqs:
                if not req.finished() and (request.abort_all or req.rid.startswith(request.rid)):
                    req.to_abort = True


def install(cls, *, timing_metadata=True):
    """Time all model stages, returning accumulated service with sampled tokens."""
    import torch
    original_init = cls.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.pp_size == 1:
            return
        self._autellix_incoming_seconds = 0.
        self._autellix_completed_seconds = 0.
        original_recv = self.pp_group.recv_tensor_dict
        original_send = self.pp_group.send_tensor_dict
        runner = self.tp_worker.model_runner
        original_forward = runner.forward
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        def recv(*args, **kwargs):
            tensors = original_recv(*args, **kwargs)
            seconds = tensors.pop("autellix_gpu_seconds", None)
            if seconds is not None:
                value = seconds if timing_metadata else seconds.item()
                if "next_token_ids" in tensors:
                    self._autellix_completed_seconds = value
                    # Non-first stages forward this previous microbatch's outputs.
                    tensors["autellix_gpu_seconds"] = seconds
                else:
                    self._autellix_incoming_seconds = value
            return tensors

        def forward(*args, **kwargs):
            start.record()
            output = original_forward(*args, **kwargs)
            end.record()
            end.synchronize()
            self._autellix_model_seconds = self._autellix_incoming_seconds + start.elapsed_time(end) / 1000
            return output

        def send(tensors, *args, **kwargs):
            if "next_token_ids" not in tensors or self.pp_group.is_last_rank:
                seconds = self._autellix_model_seconds
                # SGLang already sends Python scalars with tensor metadata.
                # The elapsed time is on the host and needs no NCCL payload.
                tensors["autellix_gpu_seconds"] = (seconds if timing_metadata else
                                                   torch.tensor(seconds, dtype=torch.float64, device=self.device))
            return original_send(tensors, *args, **kwargs)

        self.pp_group.recv_tensor_dict = recv
        self.pp_group.send_tensor_dict = send
        runner.forward = forward

    cls.__init__ = init
