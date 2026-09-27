import time
from queue import Queue, Full
from threading import Event, Thread

import numpy as np
import torch


class BatchStream:
    def __init__(self, replay, seed, count, depth, device):
        self.replay = replay
        self.seed = seed
        self.count = count
        self.device = torch.device(device)
        self.queue = Queue(maxsize=depth)
        self.stopped = Event()
        self.worker = Thread(target=self.produce, name='replay-prefetch')
        self.remaining = count
        self.prepare_seconds = 0.0

    def __enter__(self):
        self.worker.start()
        return self

    def __exit__(self, *args):
        self.stopped.set()
        self.worker.join()

    def put(self, item):
        while not self.stopped.is_set():
            try:
                self.queue.put(item, timeout=0.05)
                return
            except Full:
                pass

    def produce(self):
        try:
            rng = np.random.default_rng(self.seed)
            for _ in range(self.count):
                if self.stopped.is_set():
                    break
                started = time.monotonic()
                arrays = self.replay.sample(rng)
                tensors = [torch.from_numpy(array) for array in arrays]
                if self.device.type == 'cuda':
                    tensors = [tensor.pin_memory() for tensor in tensors]
                self.prepare_seconds += time.monotonic() - started
                self.put(tensors)
        except BaseException as error:
            self.put(error)

    def __iter__(self):
        return self

    def __next__(self):
        if self.remaining == 0:
            raise StopIteration
        batch = self.queue.get()
        if isinstance(batch, BaseException):
            raise batch
        self.remaining -= 1
        return batch
