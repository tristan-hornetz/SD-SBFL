import inspect
from types import FrameType, FunctionType
from typing import Any, Set, Tuple

from .debuggingbook.StatisticalDebugger import CoverageCollector


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
        super(ExtendedCoverageCollector, self).__init__(*args, **kwargs)
        with open(inspect.getfile(self.__init__).split("/TestWrapper/")[0] + "/TestWrapper/work_dir.info", "rt") as f:
            self.work_dir_base = str(f.readline().replace("\n", ""))

    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        """
        Same as CoverageCollector::collect, but with a more elaborate method of filtering out unimportant events
        """

        # Exclude events from file paths with these substrings:
        to_exclude = ["/TestWrapper/", "/test_", "_test.py", "/WrapClass.py"]
        name = frame.f_code.co_name
        function = self.search_func(name, frame)

        if function is None:
            function = self.create_function(frame)

        # ONLY collect functions, no other garbage
        if not isinstance(function, FunctionType):
            return

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
            return

        for s in to_exclude:
            if s in function_filename:
                return

        location = (function, frame.f_lineno)
        self._coverage.add(location)

    def events(self) -> Set[Tuple[str, int]]:
        return {((f"{inspect.getfile(func)}[{func.__name__}]" if isinstance(func,
                                                                            FunctionType) else get_file_resistant(
            func) + "[<unknown>]"),
                 lineno) for func, lineno in self._coverage}


collector_type = ExtendedCoverageCollector
