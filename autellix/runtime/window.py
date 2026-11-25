"""Freeze policy order for N decode steps, with resident reserve refill."""
from heapq import nsmallest


class SchedulingWindow:
    def __init__(self, controller, capacity, refill_idle_slots=False, resident_reserve_only=False):
        self.controller, self.capacity = controller, capacity
        self.refill_idle_slots = refill_idle_slots
        self.resident_reserve_only = resident_reserve_only
        self.capacity_limited = False
        self.remaining = 0
        self.cohort = []
        self.active = set()
        self.reserve = set()

    def select(self, available, resident, *, new_admissions=None):
        ctl = self.controller
        available = set(available)
        if not available:
            self.cohort, self.active, self.reserve, self.remaining = [], set(), set(), 0
            return set(), set()
        resident = set(resident)
        cohort = [rid for rid in self.cohort if rid in available]
        boundary = self.remaining == 0 or not cohort

        def choose(count, candidates, retained=()):
            if new_admissions is not None:
                waiting, slots = new_admissions
                budget = slots - len(waiting.intersection(retained))
                fresh = nsmallest(budget, candidates & waiting, key=ctl.key)
                candidates = (candidates - waiting).union(fresh)
            return nsmallest(count, candidates, key=ctl.key)

        if boundary:
            ctl.refresh()
            reserve_size = ctl.config.overprovision
            if self.resident_reserve_only:
                cohort = choose(self.capacity, available)
                cohort.extend(nsmallest(reserve_size, (available & resident).difference(cohort), key=ctl.key))
            else:
                cohort = choose(self.capacity + reserve_size, available)
            self.remaining = ctl.config.schedule_interval
            self.capacity_limited = False
            ctl.emit("schedule_window", requests=cohort, decode_steps=self.remaining)
        elif (self.refill_idle_slots and len(cohort) < self.capacity
              and (not self.capacity_limited or len(cohort) < len(self.cohort))):
            spare = self.capacity - len(cohort)
            refill = choose(spare, available.difference(cohort), cohort)
            if refill:
                # A completion can free KV after a failed admission. Retry
                # only after that change and preserve the surviving order.
                cohort.extend(refill)
                ctl.emit("refill_idle", requests=refill, remaining_steps=self.remaining)
        active = set(cohort[:self.capacity])
        reserve = set(cohort[self.capacity:])
        if not boundary:
            for rid in active & self.reserve & resident:
                ctl.emit("refill", rid=rid, remaining_steps=self.remaining)
        self.cohort, self.active, self.reserve = cohort, active, reserve
        # Precompute the reserve's prefill / restore its swapped KV once; hold
        # it resident thereafter. Only capacity requests participate in decode.
        prepare = reserve - resident
        return active | prepare, reserve & resident

    def can_continue_decode(self, running, prepared):
        return (self.remaining > 0 and bool(running) and running == self.active
                and self.reserve.issubset(prepared)
                and (len(running) == self.capacity or self.capacity_limited or not self.refill_idle_slots))

    def executed(self, is_prefill=False):
        if not is_prefill:
            self.remaining = max(0, self.remaining - 1)

    def drop_reserve(self):
        self.controller.emit("release_reserve", requests=list(self.reserve))
        self.cohort = self.cohort[:self.capacity]
        self.reserve = set()

    def retain_prefix(self, length):
        """Algorithm 1's first non-fitting request ends admission this window."""
        self.capacity_limited = True
        self.cohort = self.cohort[:length]
        self.active = set(self.cohort[:self.capacity])
        self.reserve = set(self.cohort[self.capacity:])
        self.controller.emit("capacity_limit", requests=list(self.cohort))
