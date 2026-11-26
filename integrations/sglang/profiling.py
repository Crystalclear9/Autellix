"""Opt-in CPU profiling of the actual GPU scheduler process."""
import cProfile
import os
from pathlib import Path


def install(cls):
    directory = os.environ.get("AUTELLIX_PROFILE_DIR")
    if not directory:
        return
    Path(directory).mkdir(parents=True, exist_ok=True)
    original_init, original_process = cls.__init__, cls.process_batch_result

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._cpu_profile = cProfile.Profile()
        self._cpu_profile.enable()

    def process(self, batch, result, *args, **kwargs):
        requests = list(batch.reqs or ())
        value = original_process(self, batch, result, *args, **kwargs)
        if any(req.finished() for req in requests):
            self._cpu_profile.dump_stats(str(Path(directory) / f"scheduler-{os.getpid()}.prof"))
            self._cpu_profile.enable()
        return value

    cls.__init__, cls.process_batch_result = init, process
