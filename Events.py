import inspect
from abc import abstractmethod
from types import FrameType, FunctionType
from typing import Any, Iterable, Iterator, Hashable

def get_file_resistant(o):
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
            if self.collector not in self.shared_coverage[o]:
                self.length += 1
            self.shared_coverage[o].add(self.collector)
        else:
            self.shared_coverage[o] = {self.collector}
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
        self.to_exclude = ["/TestWrapper/", "/test_", "_test.py", "/WrapClass.py", "/.pyenv/"]
        self.buffered_functions = [(("", "", 0), None)] * 20

    @abstractmethod
    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        pass

    def check_function(self, function: FunctionType):
        """
        Get the function that should be processed for function (might be != function itself, or None)
        :param function: A function encountered by traceit
        :return: The function to be processed, or None if the given function should be excluded from collection
        """

        function_filename = get_file_resistant(function)

        # If the function is decorated, also consider wrapped function itself
        if self.collector.work_dir_base not in function_filename:
            while hasattr(function, "__wrapped__"):
                function = function.__wrapped__
                function_filename = get_file_resistant(function)
                if self.collector.work_dir_base in function_filename:
                    break

        # Don't collect function which are defined outside of our project
        if self.collector.work_dir_base not in function_filename:
            return None

        for s in self.to_exclude:
            if s in function_filename:
                return None

        return function

    def get_function_from_frame(self, frame: FrameType):
        index = 0
        for info, func in self.buffered_functions:
            if info == (frame.f_code.co_name, frame.f_code.co_filename, frame.f_code.co_firstlineno):
                if index != 0:
                    self.buffered_functions.insert(0, self.buffered_functions.pop(index))
                return func
            index += 1

        name = frame.f_code.co_name
        function = self.collector.search_func(name, frame)

        if function is None:
            function = self.collector.create_function(frame)

        # ONLY collect functions, no other garbage
        if not (isinstance(function, FunctionType) and hasattr(function, '__name__')):
            ret = None
        else:
            ret = self.check_function(function)

        self.buffered_functions.pop()
        self.buffered_functions.insert(0, ((frame.f_code.co_name, frame.f_code.co_filename, frame.f_code.co_firstlineno), ret))
        return ret


class LineCoveredEvent(DebuggerEvent):
    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        """
        Same as CoverageCollector::collect, but with a more elaborate method of filtering out unimportant events
        Function objects are translated to strings so that the functions themselves don't have to stay in memory
        """

        function = self.get_function_from_frame(frame)

        if not function:
            return

        location = (f"Covered line @ {get_file_resistant(function)}[{function.__name__}]", frame.f_lineno)
        self.container.add(location)


class ReturnValueEvent(DebuggerEvent):
    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        """
        Collect a return value
        If possible, the hash of the returned object is stored to keep the object itself out of memory
        Otherwise, store 'Unhashable'
        """

        if event != 'return':
            return

        function = self.get_function_from_frame(frame)

        if not function:
            return

        if isinstance(arg, Hashable):
            obj_representation = f"<{str(hash(arg))}> - {str(type(arg))}"
        else:
            obj_representation = f"Unhashable {str(type(arg))}"

        event_string = (f"Returned '{obj_representation}' @ {get_file_resistant(function)}[{function.__name__}]", frame.f_lineno)
        self.container.add(event_string)


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
                pass
        return comp_str

    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        """
        Collect scalar pairs for variables altered in this frame
        """

        function = self.get_function_from_frame(frame)

        if not function:
            return

        localvars = frame.f_locals.copy()
        localvars.update(frame.f_globals)

        comp_str = []

        matching_vars = set(localvars.keys()).intersection(self.scalars.keys())
        for v in matching_vars:
            try:
                if localvars[v] != self.scalars[v]:
                    comp_str.extend(self.get_pair_strings(v, localvars))
            except:
                pass

        non_matching_vars = set(localvars.keys()).difference(self.scalars.keys())
        for v in non_matching_vars:
            comp_str.extend(self.get_pair_strings(v, localvars))

        for s in comp_str:
            event_string = (f"Pair [{s}] @ {get_file_resistant(function)}[{function.__name__}]", frame.f_lineno)
            self.container.add(event_string)

        self.scalars = localvars
