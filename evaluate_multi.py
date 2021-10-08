import os
import queue
import signal
import sys
import itertools
from multiprocessing import Queue, Process
from typing import Collection, Iterator
from sklearn.model_selection import StratifiedShuffleSplit
from shutil import rmtree
from Evaluator.Ranking import RankingInfo, MetaRanking
from evaluate_single import THREADS
from tqdm import tqdm
from Evaluator.CodeInspection.utils import mkdirRecursive
from Evaluator.CombiningMethod import *
from Evaluator.Evaluation import Evaluation
from Evaluator.RankerEvent import *
from Evaluator.SimilarityCoefficient import *
from correlations import extend_w_event_type_specific_results, extend_w_lc_best, extract_labels

TEMP_SYMLINK_DIR = "./.temp_evaluation"

EVENT_TYPES = [LineCoveredEvent, SDBranchEvent, SDReturnValueEvent, AbsoluteReturnValueEvent,
               AbsoluteScalarValueEvent]  # , SDScalarPairEvent]

ALL_EVENT_TYPES = [LineCoveredEvent, SDBranchEvent, AbsoluteReturnValueEvent, AbsoluteScalarValueEvent, SDScalarPairEvent, SDReturnValueEvent]

SIMILARITY_COEFFICIENTS = [JaccardCoefficient, SorensenDiceCoefficient, AnderbergCoefficient, OchiaiCoefficient,
                           SimpleMatchingCoefficient, RogersTanimotoCoefficient, OchiaiIICoefficient,
                           RusselRaoCoefficient, TarantulaCoefficient]

AGGREGATORS = [max, avg, geometric_mean, harmonic_mean, quadratic_mean, median, len, sum]
AGGREGATORS_ALTERNATE = []


def get_files_recursive(dir, files: List[str]):
    for f in os.listdir(dir):
        p = f"{dir}/{f}"
        if os.path.isdir(p):
            get_files_recursive(p, files)
        else:
            files.append(p)
    return files


def create_evaluation_recursive(result_dir, similarity_coefficient, combining_method: CombiningMethod,
                                save_destination="",
                                print_results=False, num_threads=-1, save_full_rankings=False, meta_rankings=None):
    evaluation = Evaluation(similarity_coefficient, combining_method, save_full_rankings=save_full_rankings)
    files = list(set(get_files_recursive(result_dir, [])))
    if os.path.exists(TEMP_SYMLINK_DIR):
        rmtree(TEMP_SYMLINK_DIR)
    mkdirRecursive(TEMP_SYMLINK_DIR)
    for f in files:
        os.symlink(os.path.realpath(f), f"{TEMP_SYMLINK_DIR}/{os.path.basename(f)}")
    evaluation.add_directory(TEMP_SYMLINK_DIR, THREADS if num_threads < 1 else num_threads)
    rmtree(TEMP_SYMLINK_DIR)
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


def run_process_list(processes, out_queue: Queue, task_name: str = "", num_threads=-1):
    num_threads = num_threads if num_threads > 0 else os.cpu_count()
    active_processes = list()
    num_processes = len(processes)
    progress_bar = tqdm(total=num_processes, desc=task_name)
    ret = list()
    while len(processes) > 0 or len(active_processes) > 0:
        while len(active_processes) < num_threads and len(processes) > 0:
            t = processes.pop()
            t.start()
            active_processes.append(t)
        try:
            ret.append(out_queue.get(timeout=1.0))
        except queue.Empty:
            pass
        for t in active_processes:
            if not t.is_alive():
                active_processes.remove(t)
                progress_bar.update(1)
    while not out_queue.empty():
        ret.append(out_queue.get())
    assert (len(ret) <= num_processes)
    assert (out_queue.empty())
    return ret


def make_tmp_folder(result_dir):
    files = list(set(get_files_recursive(result_dir, [])))
    if os.path.exists(TEMP_SYMLINK_DIR):
        rmtree(TEMP_SYMLINK_DIR)
    mkdirRecursive(TEMP_SYMLINK_DIR)
    for f in files:
        os.symlink(os.path.realpath(f), f"{TEMP_SYMLINK_DIR}/{os.path.basename(f)}")
    return TEMP_SYMLINK_DIR


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

    def create_evaluation(self, result_dir, similarity_coefficient, combining_method, meta_rankings=None):
        evaluation = create_evaluation_recursive(result_dir, similarity_coefficient, combining_method,
                                                 print_results=True, meta_rankings=meta_rankings)
        combining_method.update_results(evaluation)
        self.evaluations.append(evaluation)

    @staticmethod
    def process_mr_file(path: str, evaluations: List[Evaluation], out_queue: Queue):
        try:
            with gzip.open(path, "rb") as f:
                mr: MetaRanking = pickle.load(f)
        except Exception as e:
            print(type(e))
            traceback.print_tb(e.__traceback__)
            print(f"Could not load {path}")
            return
        out = dict()
        for ev in evaluations:
            ranking_id = f"{mr._results.project_name}_{mr._results.bug_id}"
            with mr.rank(ev.similarity_coefficient, ev.combining_method) as ranking:
                out[ev.id] = (ranking_id, RankingInfo(ranking))
        out_queue.put(out)

    def run_task(self, task: List[Tuple[str, Any, CombiningMethod]]):
        evaluations = [Evaluation(similarity_coefficient=s, combining_method=c) for _, s, c in task]
        ev_lookup = {ev.id: ev for ev in evaluations}
        tmp_dir = make_tmp_folder(task[0][0])
        out_queue = Queue()
        processes = [Process(target=self.process_mr_file, name=filename, args=(filename, evaluations, out_queue))
                     for filename in filter(lambda p: not os.path.isdir(p),
                                            list(os.path.abspath(f"{tmp_dir}/{f}") for f in os.listdir(tmp_dir)))
                     ]
        rankings = run_process_list(processes, out_queue, task_name="Processing Meta Rankings for task")
        for d in rankings:
            for ev_id, (id, ri) in d.items():
                local_ev = ev_lookup[ev_id]
                local_ev.rankings.append(id)
                local_ev.ranking_infos.append(ri)
        for ev in evaluations:
            if len(ev.rankings) > 0:
                ev.update_averages()
        self.evaluations = evaluations
        rmtree(tmp_dir)

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
                                                                            key=lambda e: sum(
                                                                                e.fraction_top_k_accurate[k] +
                                                                                e.avg_recall_at_k[k] +
                                                                                e.avg_precision_at_k[k] for k in
                                                                                [1, 3, 5, 10]),
                                                                            reverse=True))
        len = 0
        _sum = {k: 0 for k in [1, 3, 5, 10]}
        for ev in self.evaluations:
            for ri in ev.ranking_infos:
                len += 1
                for k in _sum.keys():
                    _sum[k] += (ri.buggy_in_ranking if ri.buggy_in_ranking <= k else k) / ri.num_buggy_methods
        avgs = {k: v / len for k, v in _sum.items()}
        out += f"\n\nRecall upper bound: {avgs}\n"
        return out


def build_temp_folder(temp_folder_name:str,  results_translated_folder: str, ris, test_index):
    test_ris = ris[test_index]
    if os.path.exists(temp_folder_name):
        rmtree(temp_folder_name)
    for ri in test_ris:
        new_link = f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/{temp_folder_name}/{ri.project_name}/translated_results_{ri.project_name}_{ri.bug_id}.pickle.gz"
        old_link = f"{os.path.dirname(os.path.realpath(sys.argv[0]))}/{os.path.basename(results_translated_folder)}/{ri.project_name}/translated_results_{ri.project_name}_{ri.bug_id}.pickle.gz"
        if not os.path.exists(os.path.dirname(new_link)):
            mkdirRecursive(os.path.dirname(new_link))
        os.symlink(old_link, new_link)
    return temp_folder_name


def get_split_training_ris(base_results_file: str, results_translated_folder: str, random_state=42) -> Tuple[
    List[RankingInfo], str, List[RankingInfo]]:
    # prepare data
    base_run = EvaluationRun.load(base_results_file)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=random_state)
    ris = np.array(base_run.evaluations[0].ranking_infos)
    groups = np.array(list(ri.project_name for ri in ris))
    train_index, test_index = next(splitter.split(ris, groups))
    train_ris = ris[train_index]
    test_ris = ris[test_index]
    # build testing folder
    temp_folder_name = "_results_test"

    return train_ris, build_temp_folder(temp_folder_name, results_translated_folder, ris, test_index), list(test_ris)


def get_training_data(base_results_file: str, results_translated_folder: str) -> Tuple[
    EvaluationRun, str, List[RankingInfo]]:
    # prepare data
    base_run = EvaluationRun.load(base_results_file)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=.2)
    ris = np.array(base_run.evaluations[0].ranking_infos)
    groups = np.array(list(ri.project_name for ri in ris))
    train_index, test_index = next(splitter.split(ris, groups))

    # build training evaluation run
    train_ris = ris[train_index]
    training_r_ids = set(f"{ri.project_name}-{ri.bug_id}" for ri in train_ris)
    n_evs = [Evaluation(ev.similarity_coefficient, ev.combining_method) for ev in base_run.evaluations]
    for n_ev, o_ev in zip(n_evs, base_run.evaluations):
        # .rankings = list(sorted(filter(lambda r: f"{r.info.project_name}-{r.info.bug_id}" in training_r_ids, o_ev.rankings), key=lambda ri: f"{ri.project_name}-{ri.bug_id}"))
        n_ev.ranking_infos = list(sorted(train_ris, key=lambda ri: f"{ri.project_name}-{ri.bug_id}"))
        n_ev.update_averages()
    training_run = EvaluationRun("training_run", "results_evaluation")
    training_run.evaluations = n_evs
    # build testing folder
    temp_folder_name = "_results_test"
    test_ris = ris[test_index]

    return training_run, build_temp_folder(temp_folder_name, results_translated_folder, ris, test_index), list(test_ris)


def get_methods_from_ris(ris: Iterable[RankingInfo]) -> Tuple[List[Tuple[DebuggerMethod, List[Tuple[float, type]]]],
                                                              List[Tuple[DebuggerMethod, List[Tuple[float, type]]]]]:
    buggy_methods = list()
    non_buggy_methods = list()
    for ri in ris:
        bm = ri.buggy_methods
        for m, scores in list(ri.top_20.items())[:10]:
            check = False
            for b in bm:
                if m.__eq__(b):
                    check = True
                    buggy_methods.append((m, scores))
            if not check:
                non_buggy_methods.append((m, scores))
    return buggy_methods, non_buggy_methods


def linearize_method(method: DebuggerMethod, scores: List[Tuple[float, type]], buggy: bool):
    typed_scores = {t: [] for t in ALL_EVENT_TYPES}
    for score, t in scores:
        typed_scores[t].append(score)
    all_scores = np.array([s for s, _ in scores])
    ret = [buggy, len(method.linenos), np.max(all_scores), np.average(all_scores), np.std(all_scores)]
    for t in ALL_EVENT_TYPES:
        ret.append(max(typed_scores[t] + [0]))
        ret.append(np.average(typed_scores[t]))
    return np.nan_to_num(np.array(ret), nan=0, posinf=0)


def get_linearized_method_data(ris: Iterable[RankingInfo]):
    buggy_methods, non_buggy_methods = get_methods_from_ris(ris)
    linearized = list()
    linearized.extend(linearize_method(method, scores, True) for method, scores in buggy_methods)
    linearized.extend(linearize_method(method, scores, False) for method, scores in non_buggy_methods)
    return np.array(linearized)


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
    task_similarity_coefficients = list(
        (result_dir, s, GenericCombiningMethod(max, avg)) for s in SIMILARITY_COEFFICIENTS)

    # SIMILARITY COEFFICIENTS II
    task_similarity_coefficients2 = list(
        (result_dir, s, FilteredCombiningMethod([e], max, avg)) for s, e in
        itertools.product(SIMILARITY_COEFFICIENTS, EVENT_TYPES))

    # AGGREGATORS
    perms = itertools.permutations(AGGREGATORS, 3)
    task_aggregators = list((result_dir, OchiaiCoefficient, GenericCombiningMethod(*p)) for p in perms)

    # AGGREGATORS 2
    perms = list(set(map(lambda l: l[:l.index(make_tuple) + 1],
                         filter(lambda l: make_tuple in l, itertools.permutations(AGGREGATORS + [make_tuple], 3)))))
    task_aggregators2 = list((result_dir, OchiaiCoefficient, GenericCombiningMethod(*p)) for p in perms)

    # SIMILARITY COEFFICIENTS III
    task_similarity_coefficients3 = list(
        (result_dir, s, FilteredCombiningMethod([LineCoveredEvent, SDBranchEvent], max, avg, make_tuple)) for s in
        SIMILARITY_COEFFICIENTS)

    # SIMILARITY COEFFICIENTS IV
    task_similarity_coefficients4 = list(
        (result_dir, s, GenericCombiningMethod(max, avg, make_tuple)) for s in
        SIMILARITY_COEFFICIENTS)

    # EVENT TYPE COMBINATIONS II
    event_type_combinations2 = list()
    for i in range(len(EVENT_TYPES)):
        event_type_combinations2.extend(itertools.combinations(EVENT_TYPES, i + 1))
    event_type_combination_filters2 = [FilteredCombiningMethod(es, max, avg, make_tuple) for es in
                                       sorted(event_type_combinations2, key=lambda l: len(l))]
    task_event_type_combinations2 = list((result_dir, OchiaiCoefficient, c) for c in event_type_combination_filters2)

    # AGGREGATORS RESTRICTED
    perms_r = []
    for i in range(3):
        perms_r.extend(filter(lambda p: (i < 2 and make_tuple not in p) or (i == 2 and list(p).pop() == make_tuple),
                              itertools.permutations([avg, max, make_tuple], i + 1)))
    task_aggregators_restricted = list(
        (result_dir, OchiaiCoefficient, FilteredCombiningMethod([LineCoveredEvent, SDBranchEvent], *p)) for p in
        perms_r)

    # TASK 2 - EVENT TYPE ORDERS II
    event_type_combinations = list()
    event_type_combinations.extend(list(p) for p in itertools.permutations(EVENT_TYPES, 3))
    event_type_combination_filters = [TypeOrderCombiningMethod(es, max) for es in event_type_combinations]
    task_event_type_orders2 = list((result_dir, OchiaiCoefficient, c) for c in event_type_combination_filters)

    # WEIGHTS II
    weight_maps = [[1.0, .7, .5, .3, .2], [.5, .5, .5, .5, .5], [.2, .3, .5, .7, 1.0]]
    w2_events = [LineCoveredEvent, SDBranchEvent, AbsoluteReturnValueEvent, SDScalarPairEvent]
    weights = dict()
    for i, weight_map in enumerate(weight_maps):
        weights[i] = {w2_events[i]: weight_map[i] for i in range(len(w2_events))}
    event_type_weight_filters = [AdjustingWeightedCombiningMethod(list(ws.items()), max, avg) for ws in
                                 weights.values()]
    task_weights_2 = []
    for c in event_type_weight_filters:
        task_weights_2.extend([(result_dir, OchiaiCoefficient, c)] * 50)

    # WEIGHTS III
    weight_maps = [[0.75, 1.2, -0.7, 0.7, 0], [0.01, 0.628, 0.014, 0.4, 0], [1.4, 0.475, 0.025, 0.325, 0]]
    for i in range(len(weight_maps)):
        weight_maps[i] = list(w / max(weight_maps[i]) for w in weight_maps[i])

    w3_events = [LineCoveredEvent, SDBranchEvent, AbsoluteReturnValueEvent, SDScalarPairEvent]
    weights = dict()
    for i, weight_map in enumerate(weight_maps):
        weights[i] = {w3_events[i]: weight_map[i] for i in range(len(w3_events))}
    event_type_weight_filters = [AdjustingWeightedCombiningMethod(list(ws.items()), max, avg) for ws in
                                 weights.values()]
    task_weights_3 = []
    for c in event_type_weight_filters:
        task_weights_3.extend([(result_dir, OchiaiCoefficient, c)] * 50)

    # WEIGHTS IV
    weight_maps = [[1.0, .7, .5, .3, .2], [.5, .5, .5, .5, .5], [.2, .3, .5, .7, 1.0]]
    weights = dict()
    for i, weight_map in enumerate(weight_maps):
        weights[i] = {w2_events[i]: weight_map[i] for i in range(len(w2_events))}
    event_type_weight_filters = [AdjustingWeightedCombiningMethod(list(ws.items()), avg, max) for ws in
                                 weights.values()]
    task_weights_4 = []
    for c in event_type_weight_filters:
        task_weights_4.extend([(result_dir, OchiaiCoefficient, c)] * 25)

    # AGGREGATORS 3
    perms3 = itertools.permutations(AGGREGATORS_ALTERNATE, 2)
    task_aggregators3 = list((result_dir, OchiaiCoefficient, GenericCombiningMethod(*p)) for p in perms3)
    task_aggregators3.extend(
        (result_dir, OchiaiCoefficient, FilteredCombiningMethod([LineCoveredEvent, SDBranchEvent], *p)) for p in perms3)

    # AGGREGATORS SINGLE
    task_aggregators_single = list(
        (result_dir, OchiaiCoefficient, GenericCombiningMethod(a)) for a in set(AGGREGATORS + AGGREGATORS_ALTERNATE))
    # task_aggregators_single.extend((result_dir, OchiaiCoefficient, FilteredCombiningMethod([LineCoveredEvent, SDBranchEvent], a)) for a in set(AGGREGATORS + AGGREGATORS_ALTERNATE))

    # AGGREGATORS SP
    task_aggregators_sp = task_aggregators_single.copy()
    task_aggregators_sp.extend(list((result_dir, OchiaiCoefficient, GenericCombiningMethod(a)) for a in
                                    [(max, avg), (max, sum), (max, len), (avg, sum)]))
    task_aggregators_sp.extend(
        list((result_dir, OchiaiCoefficient, FilteredCombiningMethod([LineCoveredEvent, SDBranchEvent], a)) for a in
             [(max, avg), (max, sum), (max, len), (avg, sum)]))

    # AVERAGING COMBINER
    averager_aggregators = [(max, avg), (max, median), (max, stddev), (avg, median), (avg, stddev), (median, stddev)]
    task_averaging_combiner = list(
        (result_dir, OchiaiCoefficient, AveragingCombiningMethod(GenericCombiningMethod(*a))) for a in
        averager_aggregators)
    task_averaging_combiner.extend(list((result_dir, OchiaiCoefficient, AveragingCombiningMethod(
        FilteredCombiningMethod([LineCoveredEvent, SDBranchEvent], *a))) for a in averager_aggregators))

    # EVENT TYPES SINGLE
    task_event_types_single = list(
        (result_dir, OchiaiCoefficient, FilteredCombiningMethod([e, ], max, avg)) for e in EVENT_TYPES)

    task_test = [(result_dir, OchiaiCoefficient, FilteredCombiningMethod([LineCoveredEvent, SDBranchEvent], max, avg))]

    # CLASSIFIER
    RUN_CLASSIFIER_TEST = True  # Enable / Disable classifier test. !!! RUN TEST TASK FIRST !!!
    if RUN_CLASSIFIER_TEST:
        pre_run_file = "results_evaluation/test_task.pickle.gz"
        training_ris, test_dir, test_ris = get_split_training_ris(pre_run_file, result_dir)
        X = get_linearized_method_data(training_ris)
        x_train, labels = extract_labels(X.T, 0)
        x_train = x_train.T
        combiner_lc = TypeOrderCombiningMethod([LineCoveredEvent, SDBranchEvent, AbsoluteReturnValueEvent], max)
        classifier_c = ClassifierCombiningMethod(x_train, labels, combiner_lc, test_ris)
        compound_c = TwoStageCombiningMethod(combiner_lc, classifier_c)
        classifier_evaluation: Evaluation = create_evaluation_recursive(test_dir, OchiaiCoefficient,
                                                                        compound_c,
                                                                        "results_evaluation/classifier_ev.pickle.gz",
                                                                        num_threads=8, print_results=True)

    TASKS = {  # "basic_combining_methods": task_basic_combining_methods,
        # "event_type_combinations": task_event_type_combinations,
        # "event_type_orders": task_event_type_orders,
        # "similarity_coefficients2": task_similarity_coefficients2,
        # "aggregators": task_aggregators,#
        "test_task": task_test,
        # "aggregators2": task_aggregators2,
        # "similarity_coefficients3": task_similarity_coefficients3,
        # "similarity_coefficients4": task_similarity_coefficients4,
        # "event_type_combinations2": task_event_type_combinations2,
        # "aggregators_restricted": task_aggregators_restricted,
        # "event_type_orders2": task_event_type_orders2,
        # "weights_2": task_weights_2
        # "weights_3": task_weights_3
        # "weights_4": task_weights_4
        # "aggregators3": task_aggregators3,
        # "aggregators_single": task_aggregators_single,
        # "aggregators_sp": task_aggregators_single,
        # "averaging_combiner": task_averaging_combiner,
        # "event_types_single": task_event_types_single,
    }

    signal.signal(signal.SIGINT, interrupt_handler)

    for task_name, task in TASKS.items():
        run = EvaluationRun(task_name, output_dir)
        run.run_task(task)
        print(run)
        run.save()
