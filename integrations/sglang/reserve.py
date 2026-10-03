"""Preserve an in-flight reserve's KV before releasing its radix-cache pin."""


def release_reserves(scheduler, rids=None, *, preserve=True):
    reserves = scheduler._autellix_reserve
    selected = list(reserves) if rids is None else list(rids)
    requests = {req.rid: req for req in scheduler.waiting_queue} if preserve else {}
    for rid in selected:
        node = reserves[rid]
        if preserve and scheduler._autellix_host is not None:
            req = requests[rid]
            tokens = req._autellix_reserved_tokens
            matched = scheduler.tree_cache.match_prefix(tokens)
            if len(matched.device_indices) != len(tokens):
                raise RuntimeError("a pinned reserve lost its computed KV prefix")
            # Save first: allocation/copy failure must leave the GPU pin intact.
            scheduler._autellix_host.save(rid, tokens, matched.device_indices)
            scheduler.autellix.emit("swap_out", rid=rid, tokens=len(tokens), reason="release_reserve")
        scheduler.tree_cache.dec_lock_ref(node)
        del reserves[rid]
