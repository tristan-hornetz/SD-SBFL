from abc import abstractmethod
from typing import Any, Tuple, Set, Iterable


class RankerEvent:
    """
    Represents an event as utilized by the Evaluation Framework
    """

    def __init__(self, program_element: Any, location: Tuple[str, str, int], passed_with_event: Set[Any],
                 failed_with_event: Set[Any], total_passed: int, total_failed: int):
        """
        :param program_element: The program element for which the event occurred
        :param location: The program element's location, consisting of the filename, method name and a line number
        :param passed_with_event: A set of passing collectors containing the event
        :param failed_with_event: A set of failing collectors containing the event
        :param total_passed: The total amount of passing collectors
        :param total_passed: The total amount of failing collectors
        """
        self.event_type = str(self.__class__.__name__)
        self.program_element = program_element
        self.location = location
        self.passed_with_event = passed_with_event
        self.failed_with_event = failed_with_event
        self.total_passed = total_passed
        self.total_failed = total_failed

    @abstractmethod
    def __hash__(self):
        pass


class LineCoveredEvent(RankerEvent):
    """
    Represents a line-covered event
    """

    def __init__(self, program_element: Any, location: Any, passed_with_event: Set[Any],
                 failed_with_event: Set[Any], total_passed: int, total_failed: int):
        """
        :param program_element: The program element for which the event occurred
        :param location: The program element's location, consisting of the filename, method name and a line number
        :param passed_with_event: A set of passing collectors containing the event
        :param failed_with_event: A set of failing collectors containing the event
        :param total_passed: The total amount of passing collectors
        :param total_passed: The total amount of failing collectors
        """
        super(LineCoveredEvent, self).__init__(program_element, location, passed_with_event,
                                               failed_with_event, total_passed, total_failed)

    def __hash__(self):
        return hash((self.event_type, self.location))

    def __str__(self):
        return f"Covered {self.location}"


class SDBranchEvent(RankerEvent):
    """
    Represents an SD-like branch event
    """

    def __init__(self, program_element: Any, location: Any, passed_with_event: Set[Any],
                 failed_with_event: Set[Any], total_passed: int, total_failed: int,
                 outcome: bool):
        """
        :param program_element: The program element for which the event occurred
        :param location: The program element's location, consisting of the filename, method name and a line number
        :param passed_with_event: A set of passing collectors containing the event
        :param failed_with_event: A set of failing collectors containing the event
        :param total_passed: The total amount of passing collectors
        :param total_passed: The total amount of failing collectors
        :param outcome: The outcome of the branch condition
        """
        super(SDBranchEvent, self).__init__(program_element, location, passed_with_event,
                                            failed_with_event, total_passed, total_failed)
        self.outcome = outcome

    def __hash__(self):
        return hash((self.event_type, self.location, self.outcome))

    def __str__(self):
        return f"Branch @ {self.location} - {self.outcome}"


class SDReturnValueEvent(RankerEvent):
    """
    Represents an SD-like return value event
    """

    def __init__(self, program_element: Any, location: Any, passed_with_event: Set[Any],
                 failed_with_event: Set[Any], total_passed: int, total_failed: int,
                 operator: Any, outcome: bool):
        """
        :param program_element: The program element for which the event occurred
        :param location: The program element's location, consisting of the filename, method name and a line number
        :param passed_with_event: A set of passing collectors containing the event
        :param failed_with_event: A set of failing collectors containing the event
        :param total_passed: The total amount of passing collectors
        :param total_passed: The total amount of failing collectors
        :param operator: The operator utilized for comparing the return value to 0
        :param outcome: The outcome of the comparison
        """
        super(SDReturnValueEvent, self).__init__(program_element, location, passed_with_event,
                                                 failed_with_event, total_passed, total_failed)
        self.operator = operator
        self.outcome = outcome

    def __hash__(self):
        return hash((self.event_type, self.location, self.operator, self.outcome))

    def __str__(self):
        return f"Return v. {self.operator} - {self.outcome} @ {self.location}"


class SDScalarPairEvent(RankerEvent):
    """
    Represents an SD-like scalar pair event
    """

    def __init__(self, program_element: Any, location: Any, passed_with_event: Set[Any],
                 failed_with_event: Set[Any], total_passed: int, total_failed: int,
                 operator: Any, outcome: bool, operands: Tuple[str, str]):
        """
        :param program_element: The program element for which the event occurred
        :param location: The program element's location, consisting of the filename, method name and a line number
        :param passed_with_event: A set of passing collectors containing the event
        :param failed_with_event: A set of failing collectors containing the event
        :param total_passed: The total amount of passing collectors
        :param total_passed: The total amount of failing collectors
        :param operator: The operator utilized for comparing the scalar values
        :param outcome: The outcome of the comparison
        :param operands: The operands being compared
        """
        super(SDScalarPairEvent, self).__init__(program_element, location, passed_with_event,
                                                failed_with_event, total_passed, total_failed)
        self.operator = operator
        self.outcome = outcome
        self.operands = operands

    def __hash__(self):
        return hash((self.event_type, self.location, self.operands, self.operator, self.outcome))

    def __str__(self):
        return f"Pair {self.operands[0]} {self.operator} {self.operands[1]} - {self.outcome} @ {self.location}"


class AbsoluteReturnValueEvent(RankerEvent):
    """
    Represents a return value event utilizing absolute values
    """

    def __init__(self, program_element: Any, location: Any, passed_with_event: Set[Any],
                 failed_with_event: Set[Any], total_passed: int, total_failed: int,
                 value: Any):
        """
        :param program_element: The program element for which the event occurred
        :param location: The program element's location, consisting of the filename, method name and a line number
        :param passed_with_event: A set of passing collectors containing the event
        :param failed_with_event: A set of failing collectors containing the event
        :param total_passed: The total amount of passing collectors
        :param total_passed: The total amount of failing collectors
        :param value: The return value
        """
        super(AbsoluteReturnValueEvent, self).__init__(program_element, location, passed_with_event,
                                                       failed_with_event, total_passed, total_failed)
        self.value = value

    def __hash__(self):
        return hash((self.event_type, self.location, self.value))

    def __str__(self):
        return f"Return v. {self.value} @ {self.location}"


class AbsoluteScalarValueEvent(RankerEvent):
    """
    Represents a scalar value event utilizing absolute values
    """

    def __init__(self, program_element: Any, location: Any, passed_with_event: Set[Any],
                 failed_with_event: Set[Any], total_passed: int, total_failed: int,
                 name: str, value: Any):
        """
        :param program_element: The program element for which the event occurred
        :param location: The program element's location, consisting of the filename, method name and a line number
        :param passed_with_event: A set of passing collectors containing the event
        :param failed_with_event: A set of failing collectors containing the event
        :param total_passed: The total amount of passing collectors
        :param total_passed: The total amount of failing collectors
        :param value: The scalar value
        """
        super(AbsoluteScalarValueEvent, self).__init__(program_element, location, passed_with_event,
                                                       failed_with_event, total_passed, total_failed)
        self.name = name
        self.value = value

    def __hash__(self):
        return hash((self.event_type, self.location, self.name, self.value))

    def __str__(self):
        return f"Scalar {self.name} <- {self.value} @ {self.location}"


class EventContainer(Iterable):
    """
    Container class to efficiently store and retrieve events
    """

    def __init__(self, bug: Tuple[str, int] = ("-", 0)):
        """
        :param bug: A tuple containing the bug's project name and ID
        """
        self.events = dict()
        self.events_by_program_element = dict()
        self.project_name, self.bug_id = bug

    def add(self, event: RankerEvent):
        """
        Add an event to the container

        :param event: The event to be added
        """
        h = hash(event)
        if h in self.events.keys():
            self.events[h].passed_with_event.update(event.passed_with_event)
            self.events[h].failed_with_event.update(event.failed_with_event)
        else:
            self.events[h] = event
            if event.program_element not in self.events_by_program_element:
                self.events_by_program_element[event.program_element] = {event}
            else:
                self.events_by_program_element[event.program_element].add(event)

    def __contains__(self, item):
        return hash(item) in self.events.keys()

    def __len__(self):
        return len(self.events.items())

    def __iter__(self):
        return iter(self.events.values())

    def get_from_program_element(self, program_element: Any) -> Set[RankerEvent]:
        """
        Retrieve the events related to the given program element

        :param program_element: The program element for which to retrieve events
        :return: A set of related events
        """
        if program_element not in self.events_by_program_element.keys():
            return set()
        return self.events_by_program_element[program_element]
