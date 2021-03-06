import inspect
import os
import random
import sys
from types import FrameType, TracebackType
from typing import Any, Set, Tuple, Type, Optional, Callable

from TestWrapper.root.Events import (
    SharedEventContainer,
    LineCoveredEvent,
    ReturnValueEvent,
    ScalarEvent,
)
from TestWrapper.root.debuggingbook.StatisticalDebugger import CoverageCollector

EVENT_TYPES = [LineCoveredEvent, ReturnValueEvent, ScalarEvent]


def get_file_resistant(o):
    try:
        if hasattr(o, "__code__"):
            return os.path.realpath(o.__code__.co_filename)
        if hasattr(o, "__func__"):
            return os.path.realpath(o.__func__.__code__.co_filename)
        return os.path.realpath(inspect.getfile(o))
    except TypeError:
        pass
    except:
        try:
            return os.path.abspath(inspect.getfile(o))
        except:
            pass
    return "<unknown>"


class SharedFunctionBuffer:
    def __init__(self):
        self.buffered_functions = [(("", "", 0), None)] * 20

    def put(self, info, func):
        self.buffered_functions.pop()
        self.buffered_functions.insert(0, (info, func))

    def get(self, info):
        index = 0
        for _info, func in self.buffered_functions:
            if info == _info:
                if index != 0:
                    self.buffered_functions.insert(
                        0, self.buffered_functions.pop(index)
                    )
                return func, True
            index += 1
        return None, False


class EventCollector(CoverageCollector):
    """
    Collector with modifications that make it more suitable for larger projects
    Does not only collect coverage info, but also other types of events
    """

    class NonFunction:
        def __init__(self, name: str, file: str):
            class NonCode:
                pass

            self.__name__ = name
            self.__code__ = NonCode()
            setattr(self.__code__, "co_filename", file)

    @staticmethod
    def get_project_name(work_dir_base):
        defined_in = inspect.getfile(EventCollector.get_project_name)
        if work_dir_base not in defined_in:
            return ""
        project_dir = (
            work_dir_base
            + "/"
            + defined_in.replace(work_dir_base + "/", "").split("/", 1)[0]
        )
        assert os.path.exists(project_dir + "/bugsinpy_id.info")
        with open(project_dir + "/bugsinpy_id.info", "rt") as f:
            id_info = f.read()
        for line in id_info.split("\n"):
            if "=" in line:
                split = line.split("=")
                if split[0] == "PROJECT_NAME":
                    return split[1]
        return ""

    def __init__(self, *args, **kwargs):
        super(EventCollector, self).__init__(*args, **kwargs)
        self.work_dir_base = "#"
        if os.path.exists(
            inspect.getfile(self.__init__).split("/TestWrapper/")[0]
            + "/TestWrapper/work_dir.info"
        ):
            try:
                with open(
                    inspect.getfile(self.__init__).split("/TestWrapper/")[0]
                    + "/TestWrapper/work_dir.info",
                    "rt",
                ) as f:
                    self.work_dir_base = os.path.realpath(
                        str(f.readline().replace("\n", ""))
                    )
            except:
                pass
        if self.work_dir_base == "#":
            try:
                self.work_dir_base = os.path.realpath(
                    os.path.dirname(inspect.getfile(EventCollector))
                    + "/../_BugsInPy/framework/bin/temp"
                )
            except:
                pass
        if self.work_dir_base == "#":
            self.work_dir_base = (
                os.path.abspath(inspect.getfile(EventCollector)).split("/temp/", 1)[0]
                + "/temp"
            )

        self.event_types = []
        self.function_buffer = SharedFunctionBuffer()
        self.to_exclude = ["/TestWrapper/", "/test_", "_test.py", "/WrapClass.py"]
        self.project_name = self.get_project_name(self.work_dir_base)

    def __int__(self):
        return hash(str(self))

    def check_function(self, function: Callable):
        """
        Get the function that should be processed for function (might be != function itself, or None)
        :param function: A function encountered by traceit
        :return: The function to be processed, or None if the given function should be excluded from collection
        """

        function_filename = get_file_resistant(function)

        # Don't collect function which are defined outside of our project
        # If the function is decorated, also consider wrapped function itself
        if self.work_dir_base not in function_filename:
            while hasattr(function, "__wrapped__"):
                function = function.__wrapped__
                function_filename = get_file_resistant(function)
            if self.work_dir_base not in function_filename:
                if (
                    f"/{self.project_name}/" in function_filename
                    and "/.pyenv/" in function_filename
                ):
                    function_filename = (
                        self.work_dir_base
                        + f"/{self.project_name}/{self.project_name}/"
                        + function_filename.split(f"/{self.project_name}/", 1)[1]
                    )
                    function = self.NonFunction(function.__name__, function_filename)
                else:
                    return None

        for s in self.to_exclude:
            if s in function_filename:
                return None

        return function

    def get_function_from_frame(self, frame: FrameType):

        ret, found = self.function_buffer.get(
            (
                frame.f_code.co_filename,
                frame.f_code.co_name,
                frame.f_code.co_firstlineno,
            )
        )

        if not found:
            function = self.search_func(frame.f_code.co_name, frame)
            if not (isinstance(function, Callable) and hasattr(function, "__name__")):
                function = self.NonFunction(
                    frame.f_code.co_name, frame.f_code.co_filename
                )
            ret = self.check_function(function)
            self.function_buffer.put(
                (
                    frame.f_code.co_filename,
                    frame.f_code.co_name,
                    frame.f_code.co_firstlineno,
                ),
                ret,
            )

        return ret

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """
        Tracing function.
        Saves the first function and calls collect().
        """
        function = self.get_function_from_frame(frame)

        if not function:
            return

        for t in self.event_types:
            t.collect(
                frame, event, arg, get_file_resistant(function), function.__name__
            )

    def __exit__(
        self, exc_tp: Type, exc_value: BaseException, exc_traceback: TracebackType
    ) -> Optional[bool]:
        """Exit the `with` block."""
        sys.settrace(self.original_trace_function)

        if self.is_internal_error(exc_tp, exc_value, exc_traceback):
            return False  # internal error
        else:
            return None  # all ok

    def events(self) -> Set[Tuple[str, int]]:
        self.function_buffer = SharedFunctionBuffer()
        return self._coverage


class SharedCoverageCollector(EventCollector):
    """
    CoverageCollector which keeps data structures shared between instances to reduce RAM usage
    """

    def __init__(self, *args, id=0, **kwargs):
        if "shared_coverage" in kwargs.keys():
            self.shared_coverage = kwargs.pop("shared_coverage")
        else:
            self.shared_coverage = dict()

        self.id = id
        if self.id == 0:
            self.id = random.randint(-(2 ** 63), 2 ** 63)
        super().__init__(*args, **kwargs)
        self._coverage = SharedEventContainer(self.shared_coverage, self)
        self.event_types = list(t(self._coverage, self) for t in EVENT_TYPES)

    def __int__(self):
        return self.id

    def __call__(self, *args, **kwargs):
        self.id += 1
        return self.__class__(
            *args, id=self.id, shared_coverage=self.shared_coverage, **kwargs
        )


collector_type = SharedCoverageCollector()
