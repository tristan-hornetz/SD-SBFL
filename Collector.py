import inspect
from types import FrameType, FunctionType
from typing import Any, Set, Tuple

from TestWrapper.root.debuggingbook.StatisticalDebugger import CoverageCollector
from TestWrapper.root.Events import SharedEventContainer, LineCoveredEvent, ReturnValueEvent, ScalarPairsEvent

EVENT_TYPES = [ScalarPairsEvent]#[LineCoveredEvent, ReturnValueEvent, ScalarPairsEvent]


class EventCollector(CoverageCollector):
    """
    Collector with modifications that make it more suitable for larger projects
    Does not only collect coverage info, but also other types of events
    """

    def __init__(self, *args, **kwargs):
        super(EventCollector, self).__init__(*args, **kwargs)
        with open(inspect.getfile(self.__init__).split("/TestWrapper/")[0] + "/TestWrapper/work_dir.info", "rt") as f:
            self.work_dir_base = str(f.readline().replace("\n", ""))

        self.ignore_types = list(filter(lambda e: isinstance(e, type), self.items_to_ignore))
        self.ignore_names = set(e.__name__ for e in filter(lambda e: hasattr(e, '__name__'), self.items_to_ignore))
        self.event_types = []

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

        for t in self.event_types:
            t.collect(frame, event, arg)

    def events(self) -> Set[Tuple[str, int]]:
        return self._coverage


class SharedCoverageCollector(EventCollector):
    """
    CoverageCollector which keeps data structures shared between instances to reduce RAM usage
    """

    def __init__(self, *args, **kwargs):
        if 'shared_coverage' in kwargs.keys():
            self.shared_coverage = kwargs.pop('shared_coverage')
        else:
            self.shared_coverage = dict()
        super().__init__(*args, **kwargs)
        self._coverage = SharedEventContainer(self.shared_coverage, self)
        self.event_types = list(t(self._coverage, self) for t in EVENT_TYPES)

    def __call__(self, *args, **kwargs):
        return self.__class__(*args, shared_coverage=self.shared_coverage, **kwargs)


collector_type = SharedCoverageCollector()
