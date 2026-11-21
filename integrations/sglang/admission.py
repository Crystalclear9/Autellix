"""Keep selected decodes resident when the whole selected prefill fits."""


def can_keep_running(scheduler, pending, selected):
    running = [r for r in scheduler.running_batch.reqs if r.rid in selected]
    if len(running) + len(pending) > scheduler.server_args.max_running_requests:
        return False
    cache = scheduler.tree_cache
    for request in pending:
        request.init_next_round_input(cache)
    prefill_tokens = sum(request.extend_input_len for request in pending)
    if prefill_tokens + len(running) >= scheduler.max_prefill_tokens:
        return False
    config = scheduler.autellix.config
    horizon = config.schedule_interval if config.implementation == "paper" else float("inf")
    future_tokens = sum(min(r.sampling_params.max_new_tokens, horizon) for r in pending)
    future_tokens += sum(min(max(0, r.sampling_params.max_new_tokens - len(r.output_ids)), horizon)
                         for r in running)
    # Match PrefillAdder's ownership accounting. Shared radix paths and
    # prefixes already locked by running requests consume their space once.
    for request in pending:
        cache.inc_lock_ref(request.last_node)
    try:
        budget = scheduler.token_to_kv_pool_allocator.available_size() + cache.evictable_size()
        return prefill_tokens + future_tokens + len(running) < budget
    finally:
        for request in reversed(pending):
            cache.dec_lock_ref(request.last_node)
