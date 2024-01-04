import multiprocessing
import threading
import queue

import os
import os.path as osp
import tempfile
import copy
from typing import List, Any, Union, Iterable, Optional
from numbers import Number
from tqdm import tqdm
from abc import abstractmethod, ABC
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class ThreadedProcessingBase(ABC):
    def __init__(self, num_workers=1, mode="wait", max_queue=-1, progress_bar=True) -> None:
        """
        ThreadedProcessingBase
        """

        self.num_workers = num_workers  # number of threads
        self.mode = mode
        self.max_queue = max_queue
        self.progress_bar = progress_bar

        self._input_queue = queue.Queue(self.max_queue)
        self._output_queue = queue.Queue(self.max_queue)
        self._outputs_buffer: List[Any] = []

        # daemon threads for continuously
        self.threads = [
            threading.Thread(target=self._worker_fn, args=(), name=f"{self.__class__.__name__}-{idx}")
            for idx in range(self.num_workers)
        ]
        self.lock = threading.Lock()

        for thread in self.threads:
            thread.setDaemon(True)
            thread.start()

    @property
    def num_workers(self):
        return getattr(self, "_num_workers", 1)

    @num_workers.setter
    def num_workers(self, value):
        value = -1 if value is None else int(value)
        if value < 0:
            self._num_workers = multiprocessing.cpu_count()
        else:
            self._num_workers = value

    @property
    def max_queue(self):
        return getattr(self, "_max_queue", -1)

    @max_queue.setter
    def max_queue(self, value):
        self._max_queue = max(-1, int(value))

    @property
    def mode(self):
        return getattr(self, "_mode", "wait")

    @mode.setter
    def mode(self, value):
        if value in ["wait", "nowait"]:
            self._mode = value
        else:
            raise ValueError(f"{self.__class__.__name__}: invalid mode {value}")

    def _worker_fn(self):
        # daemonic worker_fn
        while True:
            queue_idx, inputs = self._input_queue.get()
            outputs = self.worker(**inputs)
            if self.mode == "wait":
                self._output_queue.put((queue_idx, outputs))
            self._input_queue.task_done()

    def run(self, inputs: Iterable[Any]) -> Optional[Any]:
        if self.mode == "wait":
            # queueing inputs
            num_inputs = 0
            for queue_idx, input_item in enumerate(inputs):
                self._input_queue.put((queue_idx, input_item))
                num_inputs += 1

            output_iterator = (
                tqdm(range(num_inputs), desc=f"Processing-{self.__class__.__name__}")
                if self.progress_bar
                else range(num_inputs)
            )
            for _ in output_iterator:
                output = self._output_queue.get()
                with self.lock:
                    self._outputs_buffer.append(output)
                self._output_queue.task_done()
            # sort output by queue_idx and remove queue_idx
            outputs = [v for _, v in sorted(self._outputs_buffer, key=lambda x: x[0])]
            self._outputs_buffer.clear()
            return outputs

        elif self.mode == "nowait":
            for queue_idx, input_item in enumerate(inputs):
                self._input_queue.put((queue_idx, input_item))
            return None

    @abstractmethod
    def worker(self, **kwargs) -> Any:
        pass
