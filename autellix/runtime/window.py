"""Freeze policy order for N decode steps, with resident reserve refill."""
class SchedulingWindow:
    def __init__(self, controller, capacity):
        self.controller, self.capacity = controller, capacity
        self.remaining = 0
        self.cohort = []
        self.active = set()
        self.reserve = set()

    def select(self, available, resident):
        ctl = self.controller
        available = set(available)
        if not available:
            self.cohort, self.active, self.reserve, self.remaining = [], set(), set(), 0
            return set(), set()
        cohort = [rid for rid in self.cohort if rid in available]
        boundary = self.remaining == 0 or not cohort
        if boundary:
            ctl.refresh()
            cohort = sorted(available, key=ctl.key)[:self.capacity + ctl.config.overprovision]
            self.remaining = ctl.config.schedule_interval
            ctl.emit("schedule_window", requests=cohort, decode_steps=self.remaining)
        active = set(cohort[:self.capacity])
        reserve = set(cohort[self.capacity:])
        if not boundary:
            for rid in active & self.reserve & set(resident):
                ctl.emit("refill", rid=rid, remaining_steps=self.remaining)
        self.cohort, self.active, self.reserve = cohort, active, reserve
        # Precompute the reserve's prefill / restore its swapped KV once; hold
        # it resident thereafter. Only capacity requests participate in decode.
        prepare = reserve - set(resident)
        return active | prepare, reserve & set(resident)

    def executed(self, is_prefill=False):
        if not is_prefill:
            self.remaining = max(0, self.remaining - 1)

    def drop_reserve(self):
        self.controller.emit("release_reserve", requests=list(self.reserve))
        self.cohort = self.cohort[:self.capacity]
        self.reserve = set()
