"""Recover decode capacity without replacing program priority with length order."""


def make_decode_room(scheduler, batch):
    from .reserve import release_reserves
    if batch.check_decode_mem(scheduler.decode_mem_cache_buf_multiplier):
        return
    ctl, window = scheduler.autellix, scheduler._autellix_window
    release_reserves(scheduler)
    if window is not None:
        window.drop_reserve()
    while not batch.check_decode_mem(scheduler.decode_mem_cache_buf_multiplier):
        if len(batch.reqs) <= 1:
            raise RuntimeError("one request exceeds available GPU KV capacity even after releasing reserves")
        # Preserve the frozen plan's priority, including FIFO tie ordering.
        order = {rid: i for i, rid in enumerate(window.cohort)} if window is not None else None
        victim_index = max(range(len(batch.reqs)), key=lambda i:
                           order[batch.reqs[i].rid] if order is not None else ctl.key(batch.reqs[i].rid))
        req = batch.reqs[victim_index]
        length = batch.seq_lens.cpu().tolist()[victim_index]
        indices = batch.req_to_token_pool.req_to_token[req.req_pool_idx, :length]
        tokens = (req.origin_input_ids + req.output_ids)[:length]
        scheduler._autellix_host.save(req.rid, tokens, indices)
        ctl.emit("swap_out", rid=req.rid, tokens=len(tokens), reason="decode_capacity")
        tail = batch.req_to_token_pool.req_to_token[req.req_pool_idx, len(req.prefix_indices):length]
        batch.token_to_kv_pool_allocator.free(tail)
        batch.req_to_token_pool.free(req.req_pool_idx)
        batch.tree_cache.dec_lock_ref(req.last_node)
        req.reset_for_retract()
        batch.filter_batch(keep_indices=[i for i in range(len(batch.reqs)) if i != victim_index])
        batch.batch_is_full = False
        scheduler._extend_requests_to_queue([req], is_retracted=True)
        ctl.emit("retract", rid=req.rid, reason="decode_capacity")
        if window is not None:
            window.retain_prefix(order[req.rid])
