"""Keep selected decodes resident when the whole selected prefill fits."""


def can_keep_running(scheduler, pending, selected):
    running = [r for r in scheduler.running_batch.reqs if r.rid in selected]
    # Count whole pending prefixes, including cached tokens. This also covers
    # cache entries becoming non-evictable when native admission locks them.
    input_tokens = sum(len(r.origin_input_ids) + len(r.output_ids) for r in pending)
    future_tokens = sum(r.sampling_params.max_new_tokens for r in pending)
    future_tokens += sum(max(0, r.sampling_params.max_new_tokens - len(r.output_ids))
                         for r in running)
    budget = (scheduler.token_to_kv_pool_allocator.available_size()
              + scheduler.tree_cache.evictable_size())
    return (len(running) + len(pending) <= scheduler.server_args.max_running_requests
            and input_tokens + len(running) < scheduler.max_prefill_tokens
            and input_tokens + future_tokens + len(running) < budget)
