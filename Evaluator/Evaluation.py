import os
import gzip
import pickle

from .EventTranslation import EventProcessor, DEFAULT_TRANSLATORS
from .Ranking import MetaRanking
from .CombiningMethod import CombiningMethod


class Evaluation:
    def __init__(self, rankings, similarity_coefficient, combining_method: CombiningMethod, ks=None):
        if ks is None:
            ks = [1, 3, 5, 10]
        self.ks = ks

        self.rankings = rankings
        self.similarity_coefficient = similarity_coefficient
        self.combining_method = combining_method

        evaluation_metrics = {k: list(r.get_evaluation_metrics(k) for r in self.rankings) for k in self.ks}
        self.fraction_top_k_accurate = {k: sum(e[0] for e in evaluation_metrics[k]) / len(self.rankings) for k in self.ks}
        self.avg_recall_at_k = {k: sum(e[1] for e in evaluation_metrics[k]) / len(self.rankings) for k in self.ks}
        self.avg_precision_at_k = {k: sum(e[2] for e in evaluation_metrics[k]) / len(self.rankings) for k in self.ks}


class MetaEvaluation:
    def __init__(self, processor: EventProcessor = None):
        self.meta_rankings = []
        if processor is None:
            processor = EventProcessor(DEFAULT_TRANSLATORS)
        self.event_processor = processor

    def add_from_file(self, path):
        assert os.path.exists(path)
        with gzip.open(path) as f:
            _results = pickle.load(f)
        self.meta_rankings.append(MetaRanking(*self.event_processor.process(_results), _results))

    def add_from_directory(self, path):
        assert os.path.isdir(path)
        for filename in os.listdir(path):
            try:
                self.add_from_file(f"{str(path)}/{filename}")
            except Exception as e:
                print(e)

    def evaluate(self, similarity_coefficient, combining_method: CombiningMethod):
        rankings = list(r.rank(similarity_coefficient, combining_method) for r in self.meta_rankings)
        return Evaluation(rankings, similarity_coefficient, combining_method)



