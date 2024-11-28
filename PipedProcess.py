import queue
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
from time import time_ns
from typing import TypeVar, Generic, Optional, final

T = TypeVar("T")
I = TypeVar("I")
O = TypeVar("O")


class Pipe(Generic[T]):
    def __init__(self):
        self.queue: Queue[tuple[T, list[tuple[int, int]]]] = Queue(maxsize=1)

    def get(self) -> tuple[T, list[tuple[int, int]]]:
        return self.queue.get()

    def put(self, data: T, time_list: list[tuple[int, int]]):
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put((data, time_list))


class PipedProcess(ABC, Process, Generic[I, O]):
    def __init__(self):
        super().__init__()
        self.input_pipe: Optional[Pipe[I]] = None
        self.output_pipe: Optional[Pipe[O]] = None
        self.daemon = True

    @final
    def run(self):
        self.init()
        while True:
            input_data, time_list = self.input_pipe.get() if self.input_pipe else (None, [])
            start_time = time_ns()
            output_data = self.process(input_data)
            end_time = time_ns()
            time_list.append((start_time, end_time))
            if self.output_pipe:
                self.output_pipe.put(output_data, time_list)
            else:
                for i in range(len(time_list)):
                    s, e = map(lambda x: x / 10 ** 6, time_list[i])
                    print("process %d's runtime : %10.4fms" % (i, (e - s)), flush=True)
                    if i < len(time_list) - 1:
                        print(f"piped time          : %10.4fms" % (time_list[i + 1][0] / 10 ** 6 - e), flush=True)
                print("----------------------------------", flush=True)
                print("total processed time: %10.4fms" % ((time_list[-1][1] - time_list[0][0]) / 10 ** 6), flush=True)
                print("", flush=True)

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def process(self, input_data: I) -> O:
        pass


PP = TypeVar("PP", bound=PipedProcess)


class Pipeline:
    def __init__(self, *processes: PP):
        self.pipe: tuple[Pipe, ...] = tuple(Pipe() for _ in range(len(processes) - 1))
        self.processes: tuple[PipedProcess, ...] = processes
        for i in range(len(processes) - 1):
            processes[i].output_pipe = processes[i + 1].input_pipe = self.pipe[i]

    def start(self):
        for process in self.processes:
            process.start()
        try:
            for process in self.processes:
                process.join()
        except KeyboardInterrupt:
            for process in self.processes:
                process.kill()
