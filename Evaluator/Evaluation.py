import gzip
import os
import pickle
import queue
import time
import traceback

import numpy as np
from multiprocessing import Process, Queue
from typing import List
from tqdm import tqdm

from .CombiningMethod import CombiningMethod
from .Ranking import RankingInfo
from copy import deepcopy


class Evaluation:
    def __init__(self, similarity_coefficient, combining_method: CombiningMethod, ks=None, save_full_rankings=False):
        if ks is None:
            ks = [1, 3, 5, 10]
        self.ks = ks
        self.save_full_rankings = save_full_rankings
        self.rankings = list()
        self.ranking_infos: List[RankingInfo] = list()
        self.similarity_coefficient = similarity_coefficient
        self.combining_method = combining_method
        self.id = hash(str(self.combining_method) + self.similarity_coefficient.__name__)

        self.fraction_top_k_accurate = {k: 0.0 for k in
                                        self.ks}
        self.avg_recall_at_k = {k: 0.0 for k in self.ks}
        self.avg_precision_at_k = {k: 0.0 for k in self.ks}

    def update_averages(self):
        self.fraction_top_k_accurate = {k: np.average(list(ri.evaluation_metrics[k][0] for ri in self.ranking_infos)) for k in self.ks}
        self.avg_recall_at_k = {k: np.average(list(ri.evaluation_metrics[k][1] for ri in self.ranking_infos)) for k in self.ks}
        self.avg_precision_at_k = {k: np.average(list(ri.evaluation_metrics[k][2] for ri in self.ranking_infos)) for k in self.ks}

    def merge(self, other):
        assert set(self.rankings).isdisjoint(other.rankings)
        self.rankings.extend(other.rankings)
        self.ranking_infos.extend(other.ranking_infos)
        self.update_averages()

    @staticmethod
    def add_meta_ranking(self, mr_path: str, rqueue: Queue, save_full_rankings=False):
        try:
            with gzip.open(mr_path, "rb") as f:
                mr = pickle.load(f)
        except Exception as e:
            print(type(e))
            traceback.print_tb(e.__traceback__)
            print(f"Could not load {mr_path}")
            return
        ranking_id = f"{mr._results.project_name}_{mr._results.bug_id}"
        if ranking_id in self.rankings:
            return
        try:
            ranking = mr.rank(self.similarity_coefficient, self.combining_method)
        except Exception as e:
            print(type(e))
            print(str(e))
            traceback.print_tb(e.__traceback__)
            return
        metrics = dict()
        for k in self.ks:
            metrics[k] = ranking.get_evaluation_metrics(k)
        rqueue.put((ranking if save_full_rankings else ranking_id, metrics, RankingInfo(ranking)))

    def add_directory(self, dir_path, num_threads=-1):
        if num_threads < 1:
            num_threads = max(os.cpu_count() - 2, 1)
        rqueue = Queue(maxsize=num_threads)
        processes = [Process(target=Evaluation.add_meta_ranking, name=file_path, args=(self, file_path, rqueue, self.save_full_rankings))
                     for file_path in filter(lambda p: not os.path.isdir(p),
                                             list(os.path.abspath(f"{dir_path}/{f}") for f in os.listdir(dir_path)))]
        active_processes = []
        metrics = dict()
        progress_bar = tqdm(total=len(processes))
        while len(processes) > 0:
            while len(active_processes) < num_threads and len(processes) > 0:
                t = processes.pop()
                t.start()
                active_processes.append(t)
            try:
                res = rqueue.get(timeout=10.0)
                metrics[res[0]] = (res[1], res[2])
            except queue.Empty:
                pass
            for t in active_processes:
                if not t.is_alive():
                    active_processes.remove(t)
                    progress_bar.update(1)

        while len(active_processes) > 0:
            if not rqueue.empty():
                res = rqueue.get()
                metrics[res[0]] = (res[1], res[2])
            for t in active_processes:
                if not t.is_alive():
                    active_processes.remove(t)
                    progress_bar.update(1)

        while not rqueue.empty():
            res = rqueue.get()
            metrics[res[0]] = (res[1], res[2])
            time.sleep(.01)

        assert (rqueue.empty())
        assert (len(active_processes) == 0)
        assert (len(processes) == 0)

        for m in metrics.items():
            self.rankings.append(m[0])
            self.ranking_infos.append(m[1][1])

        if len(self.rankings) > 0:
            self.update_averages()

        if len(metrics.values()) == len(os.listdir(dir_path)):
            print(f"All objects in {dir_path} were added.")
        else:
            print(f"{len(metrics.values())} of {len(os.listdir(dir_path))} objects in {dir_path} were added.")

    def __str__(self):
        self.update_averages()
        out = f"Similarity Coefficient: {self.similarity_coefficient.__name__}\n"
        out += f"Combining Method: {type(self.combining_method).__name__}\n    {(os.linesep + '    ').join(str(self.combining_method).split(os.linesep))}\n\n"
        out += f"Top-k accuracy: {' | '.join(f'{k}: {v}' for k, v in sorted(self.fraction_top_k_accurate.items()))}\n"
        out += f"Avg. recall@k: {' | '.join(f'{k}: {v}' for k, v in sorted(self.avg_recall_at_k.items()))}\n"
        out += f"Avg. precision@k: {' | '.join(f'{k}: {v}' for k, v in sorted(self.avg_precision_at_k.items()))}\n"
        out += f"Avg. unique sus. scores in Top-10: {10.0 - (sum(r.top_10_suspiciousness_value_ties for r in self.ranking_infos)/len(self.ranking_infos)) if len(self.ranking_infos) > 0 else 0}"
        return out

