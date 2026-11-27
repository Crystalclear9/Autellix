"""Replicate stage-zero policy inputs through SGLang's pipeline control channel.

Each stage owns its layer-local KV cache. Only stage zero updates the shared
program table. Database reads and policy times travel with each microbatch so
all stages perform the same admission, preemption and restoration decisions.
"""
import copy
import json
import pickle
import time
from collections import deque


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


def compact_control_transfer(device):
    """Use one existing P2P message for steps that only carry policy clocks."""
    import torch
    import torch.distributed as dist

    header = torch.empty(4, device=device, dtype=torch.float64)

    def transfer(data, rank, group, src, dst):
        if rank == src:
            requests, entries = data
            clocks = [value for name, value in entries if name == "clock"]
            if requests or len(clocks) != len(entries) or len(clocks) > 2:
                payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                clocks = []
            else:
                payload = b""
            header.copy_(torch.tensor([len(payload), len(clocks), *(clocks + [0.] * (2 - len(clocks)))],
                                      dtype=header.dtype))
            dist.send(header, dst=dst, group=group)
            if payload:
                buf = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(device)
                dist.send(buf, dst=dst, group=group)
            return data
        dist.recv(header, src=src, group=group)
        size, count, *clocks = header.tolist()
        if size:
            buf = torch.empty(int(size), device=device, dtype=torch.uint8)
            dist.recv(buf, src=src, group=group)
            return pickle.loads(buf.cpu().numpy().tobytes())
        return [], [("clock", tick) for tick in clocks[:int(count)]]

    return transfer


def host_control_transfer(rank, channels):
    """Keep colocated pipeline control on a local FIFO between ranks."""
    incoming = channels[rank - 1][0] if rank else None
    outgoing = channels[rank][1] if rank < len(channels) else None
    for reader, writer in channels:
        if reader is not incoming:
            reader.close()
        if writer is not outgoing:
            writer.close()

    def transfer(data, rank, group, src, dst):
        if rank == src:
            outgoing.send(data)
            return data
        return incoming.recv()

    return transfer


def configure(scheduler, options):
    from autellix.runtime.policy import PolicyConfig, ProgramTable, RuntimeScheduler
    from autellix.runtime.window import SchedulingWindow
    import sglang.srt.managers.scheduler as module

    leader = scheduler.pp_rank == 0
    tape = PolicyTape(leader)
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
    channels = options.get("pipeline_control_channels")
    if channels:
        transfer_control = host_control_transfer(scheduler.pp_rank, channels)
    elif options.get("compact_pipeline_control"):
        transfer_control = compact_control_transfer(scheduler.device)
    else:
        transfer_control = module.point_to_point_pyobj

    def transfer(data, rank, group, src, dst):
        if rank == src:
            data = (data, tape.entries)
        return transfer_control(data, rank, group, src, dst)

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
        first = self.pp_group.is_first_rank
        runner = self.tp_worker.model_runner
        original_forward = runner.forward
        available = deque((torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                          for _ in range(self.pp_size if first else 1))
        pending = deque()

        def elapsed(pair):
            start, end = pair
            end.synchronize()
            return start.elapsed_time(end) / 1000

        def payload(seconds):
            return seconds if timing_metadata else torch.tensor(seconds, dtype=torch.float64, device=self.device)

        def recv(*args, **kwargs):
            tensors = original_recv(*args, **kwargs)
            seconds = tensors.pop("autellix_gpu_seconds", None)
            if seconds is None:
                self._autellix_incoming_seconds = 0.
                return tensors
            value = seconds if timing_metadata else seconds.item()
            if "next_token_ids" not in tensors:
                self._autellix_incoming_seconds = value
                return tensors
            if first:
                # Returned tokens depend on the first stage's forward, so
                # its CUDA events are ready without a wait on the send path.
                pair = pending.popleft()
                value += elapsed(pair)
                available.append(pair)
                seconds = payload(value)
            self._autellix_completed_seconds = value
            tensors["autellix_gpu_seconds"] = seconds
            return tensors

        def forward(*args, **kwargs):
            pair = available.popleft() if first else available[0]
            start, end = pair
            start.record()
            output = original_forward(*args, **kwargs)
            end.record()
            if first:
                pending.append(pair)
            return output

        def send(tensors, *args, **kwargs):
            if self.pp_group.is_last_rank or (not first and "next_token_ids" not in tensors):
                # Intermediate ranks can handle the preceding microbatch's
                # returned tokens before this forward needs to publish timing.
                seconds = self._autellix_incoming_seconds + elapsed(available[0])
                tensors["autellix_gpu_seconds"] = payload(seconds)
            return original_send(tensors, *args, **kwargs)

        self.pp_group.recv_tensor_dict = recv
        self.pp_group.send_tensor_dict = send
        runner.forward = forward

    cls.__init__ = init
