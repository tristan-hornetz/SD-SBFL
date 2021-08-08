import argparse
import gzip
import os
import pickle
from os.path import dirname, exists, abspath

from Evaluator.CodeInspection.utils import mkdirRecursive
from Evaluator.CombiningMethod import CombiningMethod, inv_avg, WeightedCombiningMethod
from Evaluator.Evaluation import Evaluation
from Evaluator.RankerEvent import SDBranchEvent, LineCoveredEvent, SDReturnValueEvent
from Evaluator.SimilarityCoefficient import OchiaiCoefficient

THREADS = os.cpu_count()


def create_evaluation(result_dir, similarity_coefficient, combining_method: CombiningMethod, save_destination="",
                      print_results=False, num_threads=-1):

    evaluation = Evaluation(similarity_coefficient, combining_method)
    evaluation.add_directory(result_dir, THREADS if num_threads < 1 else num_threads)

    if save_destination != "":
        if not exists(dirname(save_destination)):
            mkdirRecursive(abspath(dirname(save_destination)))
        if exists(save_destination):
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
    arg_parser = argparse.ArgumentParser(description='Evaluate fault localization results.')
    arg_parser.add_argument("-r", "--result_dir", required=True, type=str,
                            help="The directory containing test results")

    args = arg_parser.parse_args()
    combining_method = WeightedCombiningMethod(((LineCoveredEvent, .5), (SDBranchEvent, .1), (SDReturnValueEvent, .2)),
                                               max, inv_avg)
    similarity_coefficient = OchiaiCoefficient

    result_dir = os.path.realpath(args.result_dir)

    create_evaluation(result_dir, similarity_coefficient, combining_method, print_results=True)

