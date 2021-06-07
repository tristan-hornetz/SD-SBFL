import inspect
from types import FrameType, FunctionType
from typing import Any, Set, Tuple, Optional, Callable

from TestWrapper.root.debuggingbook.StatisticalDebugger import CoverageCollector


def get_file_resistant(o):
    try:
        return inspect.getfile(o)
    except TypeError:
        return "<unknown>"


class ExtendedCoverageCollector(CoverageCollector):
    """
    CoverageCollector with modifications that make it more suitable for larger projects
    """

    def __init__(self, *args, **kwargs):
        self.exclude_function_dict = dict()

        super(ExtendedCoverageCollector, self).__init__(*args, **kwargs)
        with open(inspect.getfile(self.__init__).split("/TestWrapper/")[0] + "/TestWrapper/work_dir.info", "rt") as f:
            self.work_dir_base = str(f.readline().replace("\n", ""))

        self.ignore_types = list(filter(lambda e: isinstance(e, type), self.items_to_ignore))
        self.ignore_names = set(e.__name__ for e in filter(lambda e: hasattr(e, '__name__'), self.items_to_ignore))
        # Exclude events from file paths with these substrings:
        self.to_exclude = ["/TestWrapper/", "/test_", "_test.py", "/WrapClass.py"]


    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """
        Tracing function.
        Saves the first function and calls collect().
        """
        if self.ignore_items:
            if frame.f_code.co_name in self.ignore_names:
                return
            if 'self' in frame.f_locals:
                for t in self.ignore_types:
                    if isinstance(frame.f_locals['self'], t):
                        return

        if self._function is None and event == 'call':
            # Save function
            self._function = self.create_function(frame)
            self._args = frame.f_locals.copy()
            self._argstring = ", ".join([f"{var}={repr(self._args[var])}"
                                         for var in self._args])

        self.collect(frame, event, arg)

    def exclude_function(self, function):
        """
        Should the given function be excluded from collection?
        :param function: A function encountered by traceit
        :return: True if the given function should be excluded from collection
        """

        if function in self.exclude_function_dict.keys():
            return self.exclude_function_dict[function]

        # ONLY collect functions, no other garbage
        if not isinstance(function, FunctionType):
            self.exclude_function_dict[function] = True
            return True

        function_filename = get_file_resistant(function)

        # If the function is decorated, also consider wrapped function itself
        if self.work_dir_base not in function_filename:
            while hasattr(function, "__wrapped__"):
                function = function.__wrapped__
                function_filename = get_file_resistant(function)
                if self.work_dir_base in function_filename:
                    break

        # Don't collect function which are defined outside of our project
        if self.work_dir_base not in function_filename:
            self.exclude_function_dict[function] = True
            return True

        for s in self.to_exclude:
            if s in function_filename:
                self.exclude_function_dict[function] = True
                return True

        self.exclude_function_dict[function] = False
        return False

    def search_func(self, name: str, frame: Optional[FrameType] = None) -> \
        Optional[Callable]:
        """Search in callers for a definition of the function `name`"""
        if frame is None:
            frame = self.caller_frame()

        try:
            item = None
            if name in frame.f_globals:
                item = frame.f_globals[name]
            if name in frame.f_locals:
                item = frame.f_locals[name]
            if item and callable(item):
                return item
        except Exception:
            pass
        return None

    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        """
        Same as CoverageCollector::collect, but with a more elaborate method of filtering out unimportant events
        """

        name = frame.f_code.co_name
        function = self.search_func(name, frame)

        if function is None:
            function = self.create_function(frame)

        if self.exclude_function(function):
            return

        location = (function, frame.f_lineno)
        self._coverage.add(location)

    def events(self) -> Set[Tuple[str, int]]:
        return {((f"{inspect.getfile(func)}[{func.__name__}]" if isinstance(func,
                                                                            FunctionType) else get_file_resistant(
            func) + "[<unknown>]"),
                 lineno) for func, lineno in self._coverage}


collector_type = ExtendedCoverageCollector
