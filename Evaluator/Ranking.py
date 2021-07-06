from typing import Iterable, Dict
from .RankerEvent import EventContainer
from .CombiningMethod import CombiningMethod


class Ranking(Iterable):
    def __init__(self, events: EventContainer, method_objects: Dict, similarity_coefficient, combining_method: CombiningMethod):
        self.events = list()
        for e in events:
            self.events.append((e, similarity_coefficient.compute(e)))
        self.events.sort(key=lambda v: v[1], reverse=True)
        self.ranking = list()
        for element in set(method_objects.values()):
            self.ranking.append((element, combining_method.combine(element, events, self.events)))
        self.ranking.sort(key=lambda v: v[1], reverse=True)

    def __iter__(self):
        return iter(self.ranking)
