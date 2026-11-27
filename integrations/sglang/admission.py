"""Keep selected decodes resident when higher-priority prefills fit."""


def can_keep_running(scheduler, pending, selected, *, order=None, prefill_frontier=False):
    running = [r for r in scheduler.running_batch.reqs if r.rid in selected]
    limit_to_forward = False
    if running and order is not None and (scheduler.pp_size == 1 or prefill_frontier):
        host = scheduler._autellix_host
        prepared = any(r.rid in scheduler._autellix_reserve or (host is not None and r.rid in host.saved)
                       for r in pending)
        if not prepared:
            # Fresh requests behind every retained decode can wait for KV.
            # Native admission considers them after the higher-priority prefix.
            last_running = max(order[r.rid] for r in running)
            pending = [r for r in pending if order[r.rid] < last_running]
            limit_to_forward = prefill_frontier and not scheduler.is_mixed_chunk
    if not limit_to_forward and len(running) + len(pending) > scheduler.server_args.max_running_requests:
        return False
    cache = scheduler.tree_cache
    chunk_budget = getattr(scheduler, "chunked_prefill_size", None)
    if limit_to_forward:
        # Native split prefill stops when the first chunk fills this forward.
        # Still charge the entire prompt of the final admitted request: native
        # admission requires its full request footprint before chunking it.
        frontier = []
        prefill_tokens = 0
        slots = scheduler.get_num_allocatable_reqs(len(running)) if scheduler.pp_size > 1 else len(pending)
        if pending and slots <= 0:
            return False
        for request in sorted(pending, key=lambda req: order[req.rid]):
            request.init_next_round_input(cache)
            if frontier and request.extend_input_len >= scheduler.max_prefill_tokens - prefill_tokens:
                break
            frontier.append(request)
            prefill_tokens += request.extend_input_len
            if len(frontier) >= slots or (chunk_budget is not None and 0 < chunk_budget <= prefill_tokens):
                break
        pending = frontier
        if len(running) + len(pending) > scheduler.server_args.max_running_requests:
            return False
    else:
        for request in pending:
            request.init_next_round_input(cache)
        prefill_tokens = sum(request.extend_input_len for request in pending)
    mixed_decode = len(running) if scheduler.is_mixed_chunk else 0
    # A forward can defer some prefills while keeping the selected KV on GPU.
    if chunk_budget is not None and 0 < chunk_budget <= mixed_decode:
        return False
    if scheduler.pp_size > 1 and not limit_to_forward and prefill_tokens + len(running) >= scheduler.max_prefill_tokens:
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


def fitting_decode_prefix(scheduler, pending, selected, lower_running, order, *, prefill_frontier=False):
    """Keep the longest lower-priority decode prefix that fits behind prefills."""
    lower_running = sorted(lower_running, key=order.__getitem__)
    fixed = selected.difference(lower_running)
    # The full set has already failed the budget check. Extending the prefix
    # only adds decode headroom and can bring forward more pending prefills.
    fits, fails = 0, len(lower_running)
    while fails - fits > 1:
        size = (fits + fails) // 2
        trial = fixed.union(lower_running[:size])
        if can_keep_running(scheduler, pending, trial, order=order, prefill_frontier=prefill_frontier):
            fits = size
        else:
            fails = size
    return fixed.union(lower_running[:fits])
