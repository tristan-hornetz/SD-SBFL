import inspect
from abc import abstractmethod
from types import FrameType, FunctionType
from typing import Any, Iterable, Iterator, Hashable, Callable


def get_file_resistant(o):
    if hasattr(o, '__code__'):
        return o.__code__.co_filename
    if hasattr(o, '__func__'):
        return o.__func__.__code__.co_filename
    try:
        return inspect.getfile(o)
    except TypeError:
        return "<unknown>"


class SharedEventContainer(Iterable):
    def __init__(self, shared_coverage, collector):
        self.collector = collector
        self.shared_coverage = shared_coverage
        self.length = 0

    def add(self, o):
        if o in self.shared_coverage.keys():
            id = int(self.collector)
            if id not in self.shared_coverage[o]:
                self.length += 1
                self.shared_coverage[o].add(id)
        else:
            self.shared_coverage[o] = {int(self.collector)}
            self.length += 1

    def __contains__(self, item):
        if item in self.shared_coverage.keys():
            return self.collector in self.shared_coverage[item]
        return False

    def __len__(self):
        return self.length

    def __iter__(self) -> Iterator:
        return filter(lambda k: self.collector in self.shared_coverage[k], self.shared_coverage.keys())


class DebuggerEvent:
    def __init__(self, container: SharedEventContainer, collector):
        self.container = container
        self.collector = collector
        # Exclude events from file paths with these substrings:

    @abstractmethod
    def collect(self, frame: FrameType, event: str, arg: Any, function: Callable) -> None:
        pass



class LineCoveredEvent(DebuggerEvent):
    def collect(self, frame: FrameType, event: str, arg: Any, function: Callable) -> None:
        """
        Same as CoverageCollector::collect, but with a more elaborate method of filtering out unimportant events
        Function objects are translated to strings so that the functions themselves don't have to stay in memory
        """

        event_tuple = (get_file_resistant(function), function.__name__, frame.f_lineno, "Covered")
        self.container.add(event_tuple)


class ReturnValueEvent(DebuggerEvent):
    def collect(self, frame: FrameType, event: str, arg: Any, function: Callable) -> None:
        """
        Collect a return value
        If possible, the hash of the returned object is stored to keep the object itself out of memory
        Otherwise, store 'Unhashable'
        """

        if event != 'return':
            return

        try:
            h = hash(arg)
        except:
            h = 0

        event_tuple = (get_file_resistant(function), function.__name__, frame.f_lineno, "Return", h, str(type(arg)))
        self.container.add(event_tuple)


class ScalarPairsEvent(DebuggerEvent):
    def __init__(self, *args, **kwargs):
        super(ScalarPairsEvent, self).__init__(*args, **kwargs)
        self.scalars = dict()

    def get_pair_strings(self, var: str, vars: dict):
        comp_str = []
        for ref in vars.keys():
            if ref == var:
                continue
            try:
                comp_str.append(f"{var} < {ref}: {vars[var] < vars[ref]}")
                comp_str.append(f"{var} == {ref}: {vars[var] == vars[ref]}")
            except:
                continue
        return comp_str

    def collect(self, frame: FrameType, event: str, arg: Any, function: Callable) -> None:
        """
        Collect scalar pairs for variables altered in this frame
        """

        localvars = frame.f_locals.copy()
        localvars.update(frame.f_globals)

        comp_str = []

        local_keys = set(localvars.keys())
        matching_vars = local_keys.intersection(self.scalars.keys())
        for v in matching_vars:
            try:
                if localvars[v] == self.scalars[v]:
                    continue
            except:
                continue
            comp_str.extend(self.get_pair_strings(v, localvars))

        non_matching_vars = local_keys.difference(matching_vars)
        for v in non_matching_vars:
            comp_str.extend(self.get_pair_strings(v, localvars))

        for s in comp_str:
            event_tuple = (get_file_resistant(function), function.__name__, frame.f_lineno, "Pair", s)
            self.container.add(event_tuple)

        self.scalars = localvars
