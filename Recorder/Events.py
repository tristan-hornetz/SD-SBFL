import inspect
from abc import abstractmethod
from types import FrameType
from typing import Any, Iterable, Iterator


class SharedEventContainer(Iterable):
    def __init__(self, shared_coverage, collector):
        self.collector = collector
        self.shared_coverage = shared_coverage
        self.length = 0

    def add(self, o):
        if o in self.shared_coverage.keys():
            id = int(self.collector)
            if id not in self.shared_coverage[o].keys():
                self.length += 1
                self.shared_coverage[o][id] = 1
            else:
                self.shared_coverage[o][id] += 1
        else:
            self.shared_coverage[o] = {int(self.collector): 1}
            self.length += 1

    def __contains__(self, item):
        if item in self.shared_coverage.keys():
            return int(self.collector) in self.shared_coverage[item].keys()
        return False

    def __len__(self):
        return self.length

    def __iter__(self) -> Iterator:
        return filter(lambda k: int(self.collector) in self.shared_coverage[k].keys(), self.shared_coverage.keys())


class DebuggerEvent:
    def __init__(self, container: SharedEventContainer, collector):
        self.container = container
        self.collector = collector
        # Exclude events from file paths with these substrings:

    @abstractmethod
    def collect(self, frame: FrameType, event: str, arg: Any, filename: str, func_name: str) -> None:
        pass


class LineCoveredEvent(DebuggerEvent):
    def collect(self, frame: FrameType, event: str, arg: Any, filename: str, func_name: str) -> None:
        """
        Collect information about the line that is currently being covered.
        """

        event_tuple = (filename, func_name, frame.f_lineno, "Covered", event)
        self.container.add(event_tuple)


class ReturnValueEvent(DebuggerEvent):
    def __init__(self, *args, **kwargs):
        super(ReturnValueEvent, self).__init__(*args, **kwargs)
        self.types = {int, str, float, bool}

    def collect(self, frame: FrameType, event: str, arg: Any, filename: str, func_name: str) -> None:
        """
        Collect a return value
        If possible, the hash of the returned object is stored to keep the object itself out of memory
        Otherwise, store None
        """

        if event != 'return':
            return

        try:
            if type(arg) not in self.types:
                h = hash(arg)
            else:
                h = arg
        except:
            h = None

        event_tuple = (filename, func_name, frame.f_lineno, "Return", h, str(type(arg)))
        self.container.add(event_tuple)


class ScalarEvent(DebuggerEvent):
    def __init__(self, *args, **kwargs):
        super(ScalarEvent, self).__init__(*args, **kwargs)
        self.previous_items = set()
        self.types = {int, float, bool}

    def collect(self, frame: FrameType, event: str, arg: Any, filename: str, func_name: str) -> None:
        """
        Collect scalars that were altered in this frame
        """

        localvars = set(filter(lambda i: type(i[1]) in self.types, frame.f_locals.items()))
        # localvars.update(frame.f_globals)

        changed_values = localvars.difference(self.previous_items)

        for k, v in changed_values:
            event_tuple = (filename, func_name, frame.f_lineno, "Scalar", (k, v))
            self.container.add(event_tuple)
            for o_k, o_v in localvars:
                if o_k == k:
                    continue
                self.container.add((filename, func_name, frame.f_lineno, "Pair", (k, o_k), (v == o_v, v < o_v)))

        self.previous_items = localvars
