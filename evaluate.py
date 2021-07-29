import argparse
import os
import sys
import gzip
import pickle

from Evaluator.CombiningMethod import CombineMaxThenAvg, CombineMaxThenAvgFilter, CombineMaxThenInvAvg, CombineAvgThenMax
from Evaluator.SimilarityCoefficient import OchiaiCoefficient
from Evaluator.RankerEvent import SDBranchEvent, LineCoveredEvent, SDReturnValueEvent, SDScalarPairEvent
from Evaluator.Evaluation import MetaEvaluation

THREADS = os.cpu_count() - 2

if __name__ == "__main__":
    DEFAULT_INPUT = os.path.dirname(os.path.abspath(sys.argv[0])) + "/translated_results.pickle.gz"

    arg_parser = argparse.ArgumentParser(description='Evaluate fault localization results.')
    arg_parser.add_argument("-r", "--result_file", required=False, type=str, default=DEFAULT_INPUT,
                            help="The file conataining test results")

    args = arg_parser.parse_args()

    with gzip.open(args.result_file, "rb") as f:
        meta_evaluation = MetaEvaluation.from_me(pickle.load(f))

    print(meta_evaluation.event_processor.translators)
    combining_method = CombineMaxThenInvAvg()
    evaluation = meta_evaluation.evaluate(OchiaiCoefficient, combining_method, num_threads=THREADS)

    avg_buggy_in_ranking = sum(len(ranking.buggy_in_ranking)/len(ranking.buggy_methods) for ranking in evaluation.rankings) / len(evaluation.rankings)

    print(len(meta_evaluation.meta_rankings))
    print(len(evaluation.rankings))
    print(avg_buggy_in_ranking)
    print("\n")

    print(evaluation.fraction_top_k_accurate)
    print(evaluation.avg_recall_at_k)
    print(evaluation.avg_precision_at_k)
