import gzip
import os
import pickle, time
from multiprocessing import Process, Queue

from .CombiningMethod import CombiningMethod
from .Ranking import RankingInfo


class Evaluation:
    def __init__(self, similarity_coefficient, combining_method: CombiningMethod, ks=None):
        if ks is None:
            ks = [1, 3, 5, 10]
        self.ks = ks

        self.rankings = list()
        self.ranking_infos = list()
        self.similarity_coefficient = similarity_coefficient
        self.combining_method = combining_method

        self.evaluation_metrics = {k: [] for k in self.ks}
        self.fraction_top_k_accurate = {k: 0.0 for k in
                                        self.ks}
        self.avg_recall_at_k = {k: 0.0 for k in self.ks}
        self.avg_precision_at_k = {k: 0.0 for k in self.ks}

    def update_averages(self):
        self.fraction_top_k_accurate = {k: sum(e[0] for e in self.evaluation_metrics[k]) / len(self.rankings) for k in
                                        self.ks}
        self.avg_recall_at_k = {k: sum(e[1] for e in self.evaluation_metrics[k]) / len(self.rankings) for k in self.ks}
        self.avg_precision_at_k = {k: sum(e[2] for e in self.evaluation_metrics[k]) / len(self.rankings) for k in
                                   self.ks}

    @staticmethod
    def add_meta_ranking(self, mr_path: str, rqueue: Queue):
        try:
            with gzip.open(mr_path, "rb") as f:
                mr = pickle.load(f)
        except:
            print(f"Could not load {mr_path}")
            return
        ranking_id = f"{mr._results.project_name}_{mr._results.bug_id}"
        if ranking_id in self.rankings:
            return
        try:
            ranking = mr.rank(self.similarity_coefficient, self.combining_method)
        except:
            print(f"Error with {mr_path}")
            return
        metrics = dict()
        for k in self.ks:
            metrics[k] = ranking.get_evaluation_metrics(k)
        rqueue.put((ranking_id, metrics, RankingInfo(ranking)))

    def add_directory(self, dir_path, num_threads=-1):
        if num_threads < 1:
            num_threads = max(os.cpu_count() - 2, 1)
        rqueue = Queue(maxsize=num_threads)
        processes = [Process(target=Evaluation.add_meta_ranking, name=file_path, args=(self, file_path, rqueue))
                     for file_path in list(os.path.realpath(f"{dir_path}/{f}") for f in os.listdir(dir_path))]
        active_processes = []
        metrics = dict()
        while len(processes) > 0:
            while len(active_processes) < num_threads and len(processes) > 0:
                t = processes.pop()
                t.start()
                active_processes.append(t)
            res = rqueue.get()
            metrics[res[0]] = (res[1], res[2])
            for t in active_processes:
                if not t.is_alive():
                    active_processes.remove(t)

        while len(active_processes) > 0:
            if not rqueue.empty():
                res = rqueue.get()
                metrics[res[0]] = (res[1], res[2])
            for t in active_processes:
                if not t.is_alive():
                    active_processes.remove(t)

        while not rqueue.empty():
            res = rqueue.get()
            metrics[res[0]] = (res[1], res[2])
            rqueue.join()

        assert (rqueue.empty())
        assert (len(active_processes) == 0)
        assert (len(processes) == 0)

        for m in metrics.items():
            for k in self.ks:
                self.evaluation_metrics[k].append(m[1][0][k])
            self.rankings.append(m[0])
            self.ranking_infos.append(m[1][1])

        self.update_averages()

        if len(metrics.values()) == len(os.listdir(dir_path)):
            print(f"All objects in {dir_path} were added.")
        else:
            print(f"{len(metrics.values())} of {len(os.listdir(dir_path))} objects in {dir_path} were added.")
