"""Periodic GPU utilization and memory samples for measured inference runs."""
import subprocess
import threading
import time


def gpu_snapshot(devices=()):
    command = [
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"]
    if devices:
        command.extend(("-i", ",".join(devices)))
    output = subprocess.check_output(command, text=True)
    rows = []
    for line in output.splitlines():
        index, name, total, used, utilization = [part.strip() for part in line.split(",")]
        rows.append(dict(index=int(index), name=name, total_mib=int(total), used_mib=int(used),
                         utilization=int(utilization) if utilization.isdecimal() else None))
    return rows


class GPUMonitor:
    def __init__(self, interval=2, devices=()):
        self.interval = interval
        self.devices = tuple(devices)
        self.samples = []
        self.errors = []
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        start = time.monotonic()
        while not self.stopped.is_set():
            try:
                self.samples.append(dict(seconds=time.monotonic()-start, gpus=gpu_snapshot(self.devices)))
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
