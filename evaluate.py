import argparse
import os
from os.path import dirname, exists, abspath
import sys
import gzip
import pickle

from Evaluator.CombiningMethod import CombiningMethod, GenericCombiningMethod, avg, inv_arg, WeightedCombiningMethod
from Evaluator.SimilarityCoefficient import OchiaiCoefficient
from Evaluator.RankerEvent import SDBranchEvent, LineCoveredEvent, SDReturnValueEvent, SDScalarPairEvent
from Evaluator.Evaluation import MetaEvaluation
from Evaluator.CodeInspection.utils import mkdirRecursive

THREADS = os.cpu_count() - 2


def create_evaluation(similarity_coefficient, combining_method: CombiningMethod, save_destination="",
                      print_results=False, num_threads=-1):
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
    arg_parser.add_argument("-r", "--result_file", required=False, type=str, default=DEFAULT_INPUT,
                            help="The file conataining test results")

    args = arg_parser.parse_args()
    combining_method = WeightedCombiningMethod(((LineCoveredEvent, .5), (SDBranchEvent, .1), (SDReturnValueEvent, .2)),
                                               max, inv_arg)

    with gzip.open(args.result_file, "rb") as f:
        meta_evaluation = MetaEvaluation.from_me(pickle.load(f))

    create_evaluation(OchiaiCoefficient, combining_method, print_results=True,
                      save_destination=f"./evaluation_{meta_evaluation.meta_rankings[0]._results.project_name}.pickle.gz")
