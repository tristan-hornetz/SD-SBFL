import os
import gzip
import pickle
import sys
import time

from multiprocessing import Process, Queue
from time import sleep
from copy import deepcopy

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

    @staticmethod
    def from_me(me):
        ret = MetaEvaluation(me.event_processor)
        ret.meta_rankings = me.meta_rankings
        return ret

    def add_from_file(self, path):
        assert os.path.exists(path)
        try:
            with gzip.open(path) as f:
                _results = pickle.load(f)
            self.meta_rankings.append(MetaRanking(*self.event_processor.process(_results), _results))
            print("Succeeded " + path)
        except:
            print("Failed " + path)

    def add_from_directory(self, path):
        assert os.path.isdir(path)
        for filename in sorted(os.listdir(path)):
            self.add_from_file(f"{str(path)}/{filename}")

    @staticmethod
    def get_ranking(meta_ranking: MetaRanking, similarity_coefficient, combining_method: CombiningMethod, destination: Queue):
        c = meta_ranking
        try:
            ranking = c.rank(similarity_coefficient, combining_method)
            if len(ranking.buggy_methods) > 0:
                destination.put(ranking)
        except:
            print("Not Valid")
        finally:
            destination.close()
        #print(str(c) + " - Done")

    def evaluate(self, similarity_coefficient, combining_method: CombiningMethod, num_threads=-1):
        if num_threads < 1:
            num_threads = max(os.cpu_count() - 2, 1)
        rqueue = Queue(maxsize=num_threads)
        processes = [Process(target=MetaEvaluation.get_ranking, name=str(r), args=(r, similarity_coefficient, combining_method, rqueue)) for r in self.meta_rankings]
        active_processes = []
        rankings = list()
        while len(processes) > 0:
            while len(active_processes) < num_threads and len(processes) > 0:
                t = processes.pop()
                t.start()
                active_processes.append(t)
            rankings.append(rqueue.get())
            for t in active_processes:
                if not t.is_alive():
                    active_processes.remove(t)

        while len(active_processes) > 0:
            if not rqueue.empty():
                rankings.append(rqueue.get())
            for t in active_processes:
                if not t.is_alive():
                    active_processes.remove(t)

        time.sleep(3)
        while not rqueue.empty():
            rankings.append(rqueue.get())
            rqueue.join()

        return Evaluation(rankings, similarity_coefficient, combining_method)



