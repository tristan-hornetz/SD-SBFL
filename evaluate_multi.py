import pickle
import gzip
import os
import sys
import time

from evaluate_single import THREADS
from translate import get_subdirs_recursive
from Evaluator.CodeInspection.utils import mkdirRecursive
from Evaluator.CombiningMethod import *
from Evaluator.Evaluation import Evaluation
from Evaluator.RankerEvent import SDBranchEvent, LineCoveredEvent, SDReturnValueEvent
from Evaluator.SimilarityCoefficient import OchiaiCoefficient


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


if __name__ == "__main__":
    import argparse

    DEFAULT_OUTPUT_DIR = os.path.dirname(os.path.realpath(sys.argv[0])) + "/results_evaluation"

    arg_parser = argparse.ArgumentParser(description='Evaluate fault localization results.')
    arg_parser.add_argument("-r", "--result_dir", required=False, type=str,
                            help="The directory containing test results")
    arg_parser.add_argument("-o", "--output_dir", required=False, type=str, default=DEFAULT_OUTPUT_DIR,
                            help="The directory where output files should be stored")

    args = arg_parser.parse_args()

    combining_methods = [
        GenericCombiningMethod(max, avg),
        GenericCombiningMethod(max, inv_avg),
        GenericCombiningMethod(avg, max),
        GenericCombiningMethod(inv_avg, max),
    ]

    similarity_coefficient = OchiaiCoefficient

    result_dir = os.path.realpath(args.result_dir)
    output_dir = os.path.realpath(args.output_dir)

    for combining_method in combining_methods:
        start_time = time.time()
        create_evaluation_recursive(result_dir, similarity_coefficient, combining_method, print_results=True,
                                    save_destination=output_dir +
                                                     f"/evaluation_cmethod_{combining_method.methods[0].__name__}_" +
                                                     f"{combining_method.methods[1].__name__}.pickle.gz")
        print(f"Done in {time.time() - start_time} seconds")
