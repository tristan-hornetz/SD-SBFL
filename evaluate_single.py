import argparse
import gzip
import os
import pickle
import sys
from os.path import dirname, exists, abspath

from Evaluator.CodeInspection.utils import mkdirRecursive
from Evaluator.CombiningMethod import CombiningMethod, inv_arg, WeightedCombiningMethod
from Evaluator.Evaluation import Evaluation
from Evaluator.RankerEvent import SDBranchEvent, LineCoveredEvent, SDReturnValueEvent
from Evaluator.SimilarityCoefficient import OchiaiCoefficient

THREADS = os.cpu_count() - 2


def create_evaluation(meta_evaluation, similarity_coefficient, combining_method: CombiningMethod, save_destination="",
                      print_results=False, num_threads=-1):
    if print_results:
        print(meta_evaluation.event_processor.translators)

    evaluation = meta_evaluation.evaluate(similarity_coefficient, combining_method,
                                          num_threads=THREADS if num_threads < 1 else num_threads)

    if save_destination != "":
        if not exists(dirname(save_destination)):
            mkdirRecursive(abspath(dirname(save_destination)))
        if exists(save_destination):
            os.remove(save_destination)
        with gzip.open(save_destination, "xb") as f:
            pickle.dump(evaluation, f)

    if print_results:
        avg_buggy_in_ranking = sum(
            len(ranking.buggy_in_ranking) / len(ranking.buggy_methods) for ranking in evaluation.rankings) / len(
            evaluation.rankings)

        print(len(meta_evaluation.meta_rankings))
        print(len(evaluation.rankings))
        print(avg_buggy_in_ranking)
        print("\n")

        print(evaluation.fraction_top_k_accurate)
        print(evaluation.avg_recall_at_k)
        print(evaluation.avg_precision_at_k)


if __name__ == "__main__":
    DEFAULT_INPUT = os.path.dirname(os.path.abspath(sys.argv[0])) + "/translated_results.pickle.gz"

    arg_parser = argparse.ArgumentParser(description='Evaluate fault localization results.')
    arg_parser.add_argument("-r", "--result_dir", required=False, type=str, default=DEFAULT_INPUT,
                            help="The directory containing test results")

    args = arg_parser.parse_args()
    combining_method = WeightedCombiningMethod(((LineCoveredEvent, .5), (SDBranchEvent, .1), (SDReturnValueEvent, .2)),
                                               max, inv_arg)
    similarity_coefficient = OchiaiCoefficient

    result_dir = os.path.realpath(args.result_dir)
    evaluation = Evaluation(similarity_coefficient, combining_method)
    evaluation.add_directory(result_dir, os.cpu_count())

    print(len(evaluation.rankings))
    print("\n")

    print(evaluation.fraction_top_k_accurate)
    print(evaluation.avg_recall_at_k)
    print(evaluation.avg_precision_at_k)
