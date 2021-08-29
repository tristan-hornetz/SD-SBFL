import pickle
import gzip
import os
import signal
import subprocess
import sys
import itertools
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


def interrupt_handler():
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
            except EvaluationRun.SigIntException:
                sp = subprocess.Popen(['ps', '-opid', '--no-headers', '--ppid', str(os.getpid())], encoding='utf8',
                                      stdout=subprocess.PIPE)
                child_process_ids = [int(line) for line in sp.stdout.read().splitlines()]
                for child in child_process_ids:
                    os.kill(child, signal.SIGTERM)
                print("INTERRUPTED")
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
                                                                            key=lambda e: e.avg_recall_at_k[5],
                                                                            reverse=True))
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
    event_type_weight_filters = [FilteredCombiningMethod(ws.items(), max, avg) for ws in weights]
    task_weights_1 = list((result_dir, OchiaiCoefficient, c) for c in event_type_weight_filters)

    TASKS = {#"basic_combining_methods": task_basic_combining_methods,
             #"event_type_combinations": task_event_type_combinations,
             #"event_type_orders": task_event_type_orders,
             "weights_1": task_weights_1,
             }

    signal.signal(signal.SIGINT, interrupt_handler)

    for task_name, task in TASKS.items():
        run = EvaluationRun(task_name, output_dir)
        run.run_task(task)
        run.save()