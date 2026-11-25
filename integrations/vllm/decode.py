"""Reuse FlashAttention inputs across a stable synchronous decode window."""
from functools import wraps


def attach_decode_inputs(runner, plan, controller):
    import torch
    from vllm.attention.backends.flash_attn import FlashAttentionMetadata
    from vllm.sampling_params import SamplingType
    from vllm.utils import async_tensor_h2d

    original = runner.prepare_model_input
    cached_input = None
    previous_batch = None
    previous_lengths = None

    def decode_batch(groups):
        batch, lengths, tokens = [], [], []
        for group in groups:
            params = group.sampling_params
            if (group.is_prompt or not group.do_sample or len(group.seq_data) != 1
                    or group.lora_request or group.prompt_adapter_request or group.encoder_seq_data
                    or params.sampling_type != SamplingType.GREEDY or params.logprobs is not None
                    or params.min_tokens or params.logits_processors or params.presence_penalty
                    or params.frequency_penalty or params.repetition_penalty != 1.0):
                return None, None, None
            seq_id, data = next(iter(group.seq_data.items()))
            pages = group.block_tables[seq_id]
            if not pages:
                return None, None, None
            batch.append((group.request_id, seq_id, len(pages), pages[-1]))
            lengths.append(data.get_len())
            tokens.append(data.get_last_token_id())
        return batch, lengths, tokens

    @wraps(original)
    @torch.inference_mode()
    def prepare(seq_group_metadata_list, virtual_engine=0, finished_requests_ids=None):
        nonlocal cached_input, previous_batch, previous_lengths
        batch, lengths, tokens = decode_batch(seq_group_metadata_list)
        if (cached_input is not None and plan.continuing_decode and not finished_requests_ids
                and batch == previous_batch and
                all(length == previous + 1 for length, previous in zip(lengths, previous_lengths))):
            attention = cached_input.attn_metadata
            if not attention.use_cuda_graph or max(lengths) <= runner.max_seq_len_to_capture:
                sampled = async_tensor_h2d(tokens, torch.long, runner.device, runner.pin_memory)
                attention.advance_step(cached_input, sampled.view(-1, 1), runner.block_size,
                                       attention.num_decode_tokens, len(batch))
                attention._cached_decode_metadata = None
                cached_input.seq_lens[:len(batch)] = lengths
                previous_lengths = lengths
                controller.emit("reuse_decode_inputs", requests=[item[0] for item in batch])
                return cached_input
        model_input = original(seq_group_metadata_list, virtual_engine, finished_requests_ids)
        attention = model_input.attn_metadata
        if (batch and isinstance(attention, FlashAttentionMetadata) and model_input.input_positions.ndim == 1
                and attention.context_lens_tensor.numel() == len(batch)
                and runner.model_config.get_sliding_window() is None):
            cached_input, previous_batch, previous_lengths = model_input, batch, lengths
        else:
            cached_input = previous_batch = previous_lengths = None
        return model_input

    runner.prepare_model_input = prepare
