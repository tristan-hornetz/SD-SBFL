import pickle
import gzip
import os
import signal
import subprocess
import sys
import itertools
import traceback
from typing import Collection, Iterator

from evaluate_single import THREADS
from translate import get_subdirs_recursive
from Evaluator.CodeInspection.utils import mkdirRecursive
from Evaluator.CombiningMethod import *
from Evaluator.Evaluation import Evaluation
from Evaluator.RankerEvent import *
from Evaluator.SimilarityCoefficient import *

EVENT_TYPES = [LineCoveredEvent, SDBranchEvent, SDReturnValueEvent, SDScalarPairEvent, AbsoluteReturnValueEvent,
               AbsoluteScalarValueEvent]

SIMILARITY_COEFFICIENTS = [JaccardCoefficient, SorensenDiceCoefficient, AnderbergCoefficient, OchiaiCoefficient,
                           SimpleMatchingCoefficient, RogersTanimotoCoefficient, OchiaiIICoefficient,
                           RusselRaoCoefficient, TarantulaCoefficient]

AGGREGATORS = [max, avg, geometric_mean, harmonic_mean, quadratic_mean, median]


def create_evaluation_recursive(result_dir, similarity_coefficient, combining_method: CombiningMethod,
                                save_destination="",
                                print_results=False, num_threads=-1):
    evaluation = Evaluation(similarity_coefficient, combining_method)
    dirs = get_subdirs_recursive(result_dir)
    for dir in dirs:
        evaluation.add_directory(dir, THREADS if num_threads < 1 else num_threads)

    if save_destination != "":
        if not os.path.exists(os.path.dirname(save_destination)):
            mkdirRecursive(os.path.abspath(os.path.dirname(save_destination)))
        if os.path.exists(save_destination):
            os.remove(save_destination)
        with gzip.open(save_destination, "xb") as f:
            pickle.dump(evaluation, f)

    if print_results:
        print(len(evaluation.rankings))
        print("\n")

        print(evaluation.fraction_top_k_accurate)
        print(evaluation.avg_recall_at_k)
        print(evaluation.avg_precision_at_k)

    return evaluation


def interrupt_handler(*args, **kwargs):
    raise EvaluationRun.SigIntException


class EvaluationRun(Collection):
    class SigIntException(Exception):
        pass

    def __len__(self) -> int:
        return len(self.evaluations)

    def __iter__(self) -> Iterator:
        return iter(self.evaluations)

    def __contains__(self, __x: object) -> bool:
        return __x in self.evaluations

    def __init__(self, name, destination="."):
        self.evaluations = list()
        self.destination = os.path.realpath(destination)
        self.name = name

    def run_task(self, task: Iterable[Tuple[str, Any, CombiningMethod]]):
        i = 0
        for result_dir, similarity_coefficient, combining_method in task:
            i += 1
            print(f"{self.name}, {i}: {str(similarity_coefficient)} \n{str(combining_method)}")
            try:
                self.evaluations.append(create_evaluation_recursive(result_dir, similarity_coefficient, combining_method,
                                                                    print_results=True))
            except EvaluationRun.SigIntException as e:
                sp = subprocess.Popen(['ps', '-opid', '--no-headers', '--ppid', str(os.getpid())], encoding='utf8',
                                      stdout=subprocess.PIPE)
                child_process_ids = [int(line) for line in sp.stdout.read().splitlines()]
                for child in child_process_ids:
                    os.kill(child, signal.SIGTERM)
                print("\nINTERRUPTED")
                traceback.print_tb(e.__traceback__)
            if i % 10 == 0:
                self.save()

    def save(self):
        filename = self.destination + f"/{self.name}.pickle.gz"
        if os.path.exists(filename):
            os.remove(filename)
        with gzip.open(filename, "xb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with gzip.open(filename, "rb") as f:
            obj = pickle.load(f)
        ret = EvaluationRun(obj.name, obj.destination)
        ret.evaluations = obj.evaluations
        return ret

    def __str__(self):
        out = f"EVALUATION RUN - {self.name}\n\n"
        out += "\n--------------------------\n".join(str(e) for e in sorted(self.evaluations,
                                                                            key=lambda e: sum(e.fraction_top_k_accurate[k] + e.avg_recall_at_k[k] + e.avg_precision_at_k[k]for k in [1, 3, 5, 10]),
                                                                            reverse=True))
        len = 0
        _sum = {k: 0 for k in [1, 3, 5, 10]}
        for ev in self.evaluations:
            for ri in ev.ranking_infos:
                len += 1
                for k in _sum.keys():
                    _sum[k] += (ri.buggy_in_ranking if ri.buggy_in_ranking <= k else k) / ri.num_buggy_methods
        avgs = {k: v/len for k, v in _sum.items()}
        out += f"\n\nRecall upper bound: {avgs}\n"
        return out


if __name__ == "__main__":
    import argparse

    DEFAULT_OUTPUT_DIR = os.path.dirname(os.path.realpath(sys.argv[0])) + "/results_evaluation"

    arg_parser = argparse.ArgumentParser(description='Evaluate fault localization results.')
    arg_parser.add_argument("-r", "--result_dir", required=False, type=str,
                            help="The directory containing test results")
    arg_parser.add_argument("-o", "--output_dir", required=False, type=str, default=DEFAULT_OUTPUT_DIR,
                            help="The directory where output files should be stored")
    args = arg_parser.parse_args()
    result_dir = os.path.realpath(args.result_dir)
    output_dir = os.path.realpath(args.output_dir)

    # TASK 1 - BASIC COMBINING METHODS
    basic_combining_methods = [
        GenericCombiningMethod(max),
        GenericCombiningMethod(avg),
        GenericCombiningMethod(max, avg),
        GenericCombiningMethod(max, inv_avg),
        GenericCombiningMethod(avg, max),
        GenericCombiningMethod(avg, inv_avg),
    ]
    task_basic_combining_methods = list((result_dir, OchiaiCoefficient, c) for c in basic_combining_methods)

    # TASK 2 - EVENT TYPE ORDERS
    event_type_combinations = list()
    event_type_combinations.extend(list(p) for p in itertools.permutations(EVENT_TYPES, 4))
    event_type_combination_filters = [TypeOrderCombiningMethod(es, max, avg) for es in event_type_combinations]
    task_event_type_orders = list((result_dir, OchiaiCoefficient, c) for c in event_type_combination_filters)

    # TASK 3 - EVENT TYPE COMBINATIONS
    event_type_combinations = list()
    for i in range(len(EVENT_TYPES)):
        event_type_combinations.extend(itertools.combinations(EVENT_TYPES, i + 1))
    event_type_combination_filters = [FilteredCombiningMethod(es, max, avg) for es in event_type_combinations]
    task_event_type_combinations = list((result_dir, OchiaiCoefficient, c) for c in event_type_combination_filters)

    # TASK 4 - WEIGHTS I
    weight_map = [1.0, .7, .5, .3, .2, .1]
    weights = list()
    for p in itertools.permutations(EVENT_TYPES):
        weights.append({p[i]: weight_map[i] for i in range(len(p))})
    event_type_weight_filters = [WeightedCombiningMethod(list(ws.items()), max, avg) for ws in weights]
    task_weights_1 = list((result_dir, OchiaiCoefficient, c) for c in event_type_weight_filters)

    # SIMILARITY COEFFICIENTS
    task_similarity_coefficients = list((result_dir, s, GenericCombiningMethod(max, avg)) for s in SIMILARITY_COEFFICIENTS)

    # SIMILARITY COEFFICIENTS II
    task_similarity_coefficients2 = list(
        (result_dir, s, FilteredCombiningMethod([e], max, avg)) for s, e in itertools.product(SIMILARITY_COEFFICIENTS, EVENT_TYPES))

    # AGGREGATORS
    perms = itertools.permutations(AGGREGATORS, 3)
    task_aggregators = list((result_dir, OchiaiCoefficient, GenericCombiningMethod(*p)) for p in perms)

    # AGGREGATORS 2
    perms = list(set(map(lambda l: l[:l.index(make_tuple) + 1], filter(lambda l: make_tuple in l, itertools.permutations(AGGREGATORS + [make_tuple], 3)))))
    task_aggregators2 = list((result_dir, OchiaiCoefficient, GenericCombiningMethod(*p)) for p in perms)

    # SIMILARITY COEFFICIENTS III
    task_similarity_coefficients3 = list(
        (result_dir, s, FilteredCombiningMethod([LineCoveredEvent, SDBranchEvent], max, avg, make_tuple)) for s in SIMILARITY_COEFFICIENTS)

    # SIMILARITY COEFFICIENTS IV
    task_similarity_coefficients4 = list(
        (result_dir, s,  GenericCombiningMethod(max, avg, make_tuple)) for s in
        SIMILARITY_COEFFICIENTS)

    # EVENT TYPE COMBINATIONS II
    event_type_combinations2 = list()
    for i in range(len(EVENT_TYPES)):
        event_type_combinations2.extend(itertools.combinations(EVENT_TYPES, i + 1))
    event_type_combination_filters2 = [FilteredCombiningMethod(es, max, avg, make_tuple) for es in sorted(event_type_combinations2, key=lambda l: len(l))]
    task_event_type_combinations2 = list((result_dir, OchiaiCoefficient, c) for c in event_type_combination_filters2)

    # AGGREGATORS RESTRICTED
    perms_r = []
    for i in range(3):
        perms_r.extend(filter(lambda p: (i < 2 and make_tuple not in p) or (i == 2 and list(p).pop() == make_tuple), itertools.permutations([avg, max, make_tuple], i + 1)))
    task_aggregators_restricted = list((result_dir, OchiaiCoefficient, FilteredCombiningMethod([LineCoveredEvent, SDBranchEvent], *p)) for p in perms_r)

    task_test = [(result_dir, OchiaiCoefficient, LinPredCombiningMethod(max, avg, make_tuple)),]

    TASKS = {#"basic_combining_methods": task_basic_combining_methods,
             #"event_type_combinations": task_event_type_combinations,
             #"event_type_orders": task_event_type_orders,
             #"similarity_coefficients2": task_similarity_coefficients2,
             #"aggregators": task_aggregators,#
             "test_task": task_test
             #"aggregators2": task_aggregators2,
             #"similarity_coefficients3": task_similarity_coefficients3,
             #"similarity_coefficients4": task_similarity_coefficients4,
             #"event_type_combinations2": task_event_type_combinations2,
             #"aggregators_restricted": task_aggregators_restricted,
             }

    signal.signal(signal.SIGINT, interrupt_handler)

    for task_name, task in TASKS.items():
        run = EvaluationRun(task_name, output_dir)
        run.run_task(task)
        print(run)
        run.save()
