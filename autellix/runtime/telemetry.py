"""Periodic GPU utilization and memory samples for measured inference runs."""
import subprocess
import threading
import time


def gpu_snapshot():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in output.splitlines():
        index, name, total, used, utilization = [part.strip() for part in line.split(",")]
        rows.append(dict(index=int(index), name=name, total_mib=int(total),
                         used_mib=int(used), utilization=int(utilization)))
    return rows


class GPUMonitor:
    def __init__(self, interval=2):
        self.interval = interval
        self.samples = []
        self.errors = []
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        start = time.monotonic()
        while not self.stopped.is_set():
            try:
                self.samples.append(dict(seconds=time.monotonic()-start, gpus=gpu_snapshot()))
            except (OSError, subprocess.CalledProcessError, ValueError) as error:
                self.errors.append(str(error))
                return
            self.stopped.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stopped.set()
        self.thread.join()
