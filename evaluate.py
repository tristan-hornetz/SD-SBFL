import queue
import signal
import sys
import pickle
import gzip
from multiprocessing import Queue, Process
from typing import Collection, Iterator
from shutil import rmtree
from Evaluator.Ranking import RankingInfo, MetaRanking
from Evaluator.evaluator_utils import THREADS
from tqdm import tqdm
from Evaluator.CodeInspection.utils import mkdirRecursive
from Evaluator.CombiningMethod import *
from Evaluator.Evaluation import Evaluation
from Evaluator.RankerEvent import *
from Evaluator.SimilarityCoefficient import *
from numpy import std

TEMP_SYMLINK_DIR = os.path.realpath("./.temp_evaluation")

EVENT_TYPES = [LineCoveredEvent, SDBranchEvent, SDReturnValueEvent, AbsoluteReturnValueEvent, AbsoluteScalarValueEvent]

ALL_EVENT_TYPES = [LineCoveredEvent, SDBranchEvent, AbsoluteReturnValueEvent, AbsoluteScalarValueEvent, SDScalarPairEvent, SDReturnValueEvent]

SIMILARITY_COEFFICIENTS = [JaccardCoefficient, SorensenDiceCoefficient, AnderbergCoefficient, OchiaiCoefficient,
                           SimpleMatchingCoefficient, RogersTanimotoCoefficient, OchiaiIICoefficient,
                           RusselRaoCoefficient, TarantulaCoefficient]

AGGREGATORS = [max, avg, geometric_mean, harmonic_mean, quadratic_mean, median, len, sum]


def get_files_recursive(dir: str, files: List[str]) -> List[str]:
    """
    Recursively find files in subdirectories of dir

    :param dir: The directory to search in
    :param files: The list of detected files. Should be initialized as an empty list
    :return: A list of files found
    """
    for f in os.listdir(dir):
        p = f"{dir}/{f}"
        if os.path.isdir(p):
            get_files_recursive(p, files)
        else:
            files.append(p)
    return files


def create_evaluation_recursive(result_dir: str, similarity_coefficient, combining_method: CombiningMethod,
                                save_destination="",
                                print_results=False, num_threads=-1) -> Evaluation:
    """
    Create an evaluation from the given configuration and the translated result files in result_dir

    :param result_dir: The directory containing the translated result files, or subdirectories containing such
    :param combining_method: The combining method to be used
    :param similarity_coefficient: The similarity coefficient to be used. Can either be an instance or just the type.
    :param save_destination: Optional. If given, the evaluation is saved at the specified location
    :param num_threads: The number of parallel threads to create. Default is the number of available cores
    :return: The evaluation created
    """
    evaluation = Evaluation(similarity_coefficient, combining_method)
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


def run_process_list(processes: List[Process], out_queue: Queue, task_name: str = "", num_threads=-1) -> List[Any]:
    """
    Start the parallel execution of the processes in processes. Each process should put its output to out_queue upon termination.

    :param processes: A list of pre-initialized process instances
    :param out_queue: The queue to which the process output is added
    :param task_name: An optional name given to the process list, displayed with the progress bar
    :param num_threads: The number of parallel threads to create. Default is the number of available cores
    :return: A list containing the output of every process
    """
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
    assert (len(ret) > 0)
    assert (out_queue.empty())
    return list(sorted(ret, key=lambda e: str(e)))


def make_tmp_folder(result_dir: str) -> str:
    """
    Create a temporary folder containing symlinks to the recursively found files in result_dir

    :param result_dir: The directory containing the translated result files, or subdirectories containing such
    :return: The path to the temporary directory
    """
    files = list(set(get_files_recursive(result_dir, [])))
    if os.path.exists(TEMP_SYMLINK_DIR):
        rmtree(TEMP_SYMLINK_DIR)
    mkdirRecursive(TEMP_SYMLINK_DIR)
    for f in files:
        os.symlink(os.path.realpath(f), f"{TEMP_SYMLINK_DIR}/{os.path.basename(f)}")
    return TEMP_SYMLINK_DIR


def interrupt_handler(*args, **kwargs):
    """
    Can handle an interrupt signal to prevent termination
    """
    raise EvaluationRun.SigIntException


class EvaluationRun(Collection):
    """
    Represents a collection of multiple related evaluations
    """
    class SigIntException(Exception):
        """
        Can be raised on an interrupt signal to prevent termination
        """
        pass

    def __len__(self) -> int:
        return len(self.evaluations)

    def __iter__(self) -> Iterator:
        return iter(self.evaluations)

    def __contains__(self, __x: object) -> bool:
        return __x in self.evaluations

    def __init__(self, name: str, destination: str = "."):
        """
        :param name: The unique name of the evaluation run
        :param destination: The folder to save the run in
        """
        self.evaluations = list()
        self.destination = os.path.realpath(destination)
        self.name = name

    def create_evaluation(self, result_dir: str, similarity_coefficient, combining_method: CombiningMethod):
        """
        Create and add an evaluation based on the given configuration from the translated result files in result_dir

        :param result_dir: The directory containing the translated results
        :param combining_method: The combining method to be used
        :param similarity_coefficient: The similarity coefficient to be used. Can either be an instance or just the type.
        """
        evaluation = create_evaluation_recursive(result_dir, similarity_coefficient, combining_method,
                                                 print_results=True)
        combining_method.update_results(evaluation)
        self.evaluations.append(evaluation)

    @staticmethod
    def process_mr_file(path: str, evaluations: List[Evaluation], out_queue: Queue):
        """
        Process a single translated results file for multiple pre-initialized evaluations.
        Intended for use with multiprocessing.

        :param path: The translated result file's location
        :param evaluations: A list of pre-initialized evaluations
        :param out_queue: A Queue instance to append the resulting ranking infos to. Output type: Dict[str, Tuple[str, RankingInfo]]
        """
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
        """
        Run a task and add the resulting evaluations

        :param task: A list of tuples containing configurations of the format (<result_dir>, <similarity_coefficient>, <combining_method>)
        """
        evaluations = [Evaluation(similarity_coefficient=s, combining_method=c) for _, s, c in task]
        ev_lookup = {ev.id: ev for ev in evaluations}
        tmp_dir = make_tmp_folder(task[0][0])
        out_queue = Queue()
        processes = [Process(target=self.process_mr_file, name=filename, args=(filename, evaluations, out_queue))
                     for filename in filter(lambda p: not os.path.isdir(p),
                                            list(os.path.abspath(f"{tmp_dir}/{f}") for f in os.listdir(tmp_dir)))
                     ]
        rankings = run_process_list(processes, out_queue, task_name="Processing results for task")
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
        """
        Save the evaluation run on disk
        """
        filename = self.destination + f"/{self.name}.pickle.gz"
        if os.path.exists(filename):
            os.remove(filename)
        with gzip.open(filename, "xb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str):
        """
        Load a saved EvaluationRun instance from disk

        :param filename: The file to load
        """
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
                    _sum[k] += ((ri.buggy_in_ranking if ri.buggy_in_ranking <= k else k) / ri.num_buggy_methods) if ri.num_buggy_methods > 0 else 0
        avgs = {k: v / len for k, v in _sum.items()}
        out += f"\n\nRecall upper bound: {avgs}\n"
        return out


if __name__ == "__main__":
    import argparse

    DEFAULT_OUTPUT_DIR = os.path.dirname(os.path.realpath(sys.argv[0])) + "/results_evaluation"

    arg_parser = argparse.ArgumentParser(description='Evaluate fault localization results.')
    arg_parser.add_argument("-r", "--result_dir", required=True, type=str,
                            help="The directory containing test results")
    arg_parser.add_argument("-o", "--output_dir", required=False, type=str, default=DEFAULT_OUTPUT_DIR,
                            help="The directory where output files should be stored")
    arg_parser.add_argument("-a", "--advanced", help="Evaluate with multiple different combinations.", action='store_true')

    args = arg_parser.parse_args()
    result_dir = os.path.realpath(args.result_dir)
    output_dir = os.path.realpath(args.output_dir)

    if not os.path.exists(output_dir):
        mkdirRecursive(output_dir)

    thesis_basic = [(result_dir, OchiaiCoefficient, GenericCombiningMethod(max, avg))]

    if args.advanced:
        # EVENT TYPES SINGLE
        task_event_types_single = list(
            (result_dir, OchiaiCoefficient, FilteredCombiningMethod([e, ], max, avg)) for e in
            [LineCoveredEvent, SDBranchEvent, SDReturnValueEvent, SDScalarPairEvent, AbsoluteReturnValueEvent,
             AbsoluteScalarValueEvent])
        # SIMILARITY COEFFICIENTS SINGLE
        task_similarity_coefficients_single = list(
            (result_dir, s, GenericCombiningMethod(max, avg)) for s in SIMILARITY_COEFFICIENTS)
        # COMBINING METHODS
        selected_combining_methods = [(max,), (avg,), (max, avg), (avg, max), (max, avg, make_tuple),
                                      (avg, max, make_tuple), (max, std), (max, sum)]
        task_combining_methods_thesis = list(
            (result_dir, OchiaiCoefficient, GenericCombiningMethod(*cs)) for cs in selected_combining_methods)
        # SELECTED COMBINATIONS
        selected_combinations = [
            [LineCoveredEvent, SDBranchEvent],
            [SDBranchEvent, SDScalarPairEvent, AbsoluteReturnValueEvent],
            [SDBranchEvent, SDReturnValueEvent, SDScalarPairEvent],
            [AbsoluteScalarValueEvent, AbsoluteReturnValueEvent],
            [SDBranchEvent, SDReturnValueEvent, SDScalarPairEvent, AbsoluteScalarValueEvent, AbsoluteReturnValueEvent],
        ]
        task_selected_combinations = list(
            (result_dir, OchiaiCoefficient, FilteredCombiningMethod(es, max, avg)) for es in selected_combinations)
        selected_event_type_orders = [
            [LineCoveredEvent, SDBranchEvent, AbsoluteScalarValueEvent],
            [LineCoveredEvent, SDBranchEvent, AbsoluteScalarValueEvent, AbsoluteReturnValueEvent, SDReturnValueEvent],
            [SDBranchEvent, SDScalarPairEvent, AbsoluteReturnValueEvent],
            [AbsoluteReturnValueEvent, LineCoveredEvent, AbsoluteScalarValueEvent],
            [AbsoluteScalarValueEvent, LineCoveredEvent, AbsoluteReturnValueEvent]
        ]
        task_selected_event_type_orders = list(
            (result_dir, OchiaiCoefficient, TypeOrderCombiningMethod(es, max)) for es in selected_event_type_orders)
        tasks_in_thesis = {
            "thesis_basic": thesis_basic,
            "thesis_event_types_single": task_event_types_single,
            "thesis_similarity_coefficients": task_similarity_coefficients_single,
            "thesis_combining_methods": task_combining_methods_thesis,
            "thesis_combinations": task_selected_combinations,
            "thesis_orders": task_selected_event_type_orders,
        }
    else:
        tasks_in_thesis = {
            "thesis_basic": thesis_basic,
        }

    #task_test = [(result_dir, OchiaiCoefficient, FilteredCombiningMethod([LineCoveredEvent, SDBranchEvent], max, avg))]

    signal.signal(signal.SIGINT, interrupt_handler)

    for task_name, task in tasks_in_thesis.items():
        run = EvaluationRun(task_name, output_dir)
        run.run_task(task)
        run.save()
        print(run)
