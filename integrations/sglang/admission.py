"""Keep selected decodes resident when higher-priority prefills fit."""


def can_keep_running(scheduler, pending, selected, *, order=None):
    running = [r for r in scheduler.running_batch.reqs if r.rid in selected]
    if running and order is not None and scheduler.pp_size == 1:
        host = scheduler._autellix_host
        prepared = any(r.rid in scheduler._autellix_reserve or (host is not None and r.rid in host.saved)
                       for r in pending)
        if not prepared:
            # Fresh requests behind every retained decode can wait for KV.
            # Native admission considers them after the higher-priority prefix.
            last_running = max(order[r.rid] for r in running)
            pending = [r for r in pending if order[r.rid] < last_running]
    if len(running) + len(pending) > scheduler.server_args.max_running_requests:
        return False
    cache = scheduler.tree_cache
    for request in pending:
        request.init_next_round_input(cache)
    prefill_tokens = sum(request.extend_input_len for request in pending)
    chunk_budget = getattr(scheduler, "chunked_prefill_size", None)
    mixed_decode = len(running) if scheduler.is_mixed_chunk else 0
    # A forward can defer some prefills while keeping the selected KV on GPU.
    if chunk_budget is not None and 0 < chunk_budget <= mixed_decode:
        return False
    if scheduler.pp_size > 1 and prefill_tokens + len(running) >= scheduler.max_prefill_tokens:
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


def fitting_decode_prefix(scheduler, pending, selected, lower_running, order):
    """Keep the longest lower-priority decode prefix that fits behind prefills."""
    lower_running = sorted(lower_running, key=order.__getitem__)
    fixed = selected.difference(lower_running)
    # The full set has already failed the budget check. Extending the prefix
    # only adds decode headroom and can bring forward more pending prefills.
    fits, fails = 0, len(lower_running)
    while fails - fits > 1:
        size = (fits + fails) // 2
        trial = fixed.union(lower_running[:size])
        if can_keep_running(scheduler, pending, trial, order=order):
            fits = size
        else:
            fails = size
    return fixed.union(lower_running[:fits])
