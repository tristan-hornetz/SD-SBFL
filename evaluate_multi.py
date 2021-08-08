import pickle
import gzip
import os
import time

from evaluate_single import THREADS
from translate import get_subdirs_recursive
from Evaluator.CodeInspection.utils import mkdirRecursive
from Evaluator.CombiningMethod import CombiningMethod, inv_arg, WeightedCombiningMethod
from Evaluator.Evaluation import Evaluation
from Evaluator.RankerEvent import SDBranchEvent, LineCoveredEvent, SDReturnValueEvent
from Evaluator.SimilarityCoefficient import OchiaiCoefficient


def create_evaluation_recursive(result_dir, similarity_coefficient, combining_method: CombiningMethod, save_destination="",
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


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser(description='Evaluate fault localization results.')
    arg_parser.add_argument("-r", "--result_dir", required=False, type=str,
                            help="The directory containing test results")

    args = arg_parser.parse_args()

    combining_method = WeightedCombiningMethod(((LineCoveredEvent, .5), (SDBranchEvent, .1), (SDReturnValueEvent, .2)),
                                               max, inv_arg)
    similarity_coefficient = OchiaiCoefficient

    result_dir = os.path.realpath(args.result_dir)

    start_time = time.time()

    create_evaluation_recursive(result_dir, similarity_coefficient, combining_method, print_results=True)

    print(f"Done in {time.time() - start_time} seconds")

