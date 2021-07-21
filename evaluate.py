import argparse
import os
import sys
import gzip
import pickle

from Evaluator.CombiningMethod import CombineMaxThenAvg, CombineMaxThenAvgFilter
from Evaluator.SimilarityCoefficient import OchiaiCoefficient
from Evaluator.RankerEvent import SDBranchEvent, LineCoveredEvent, SDReturnValueEvent, SDScalarPairEvent

if __name__ == "__main__":
    DEFAULT_INPUT = os.path.dirname(os.path.abspath(sys.argv[0])) + "/translated_results.pickle.gz"

    arg_parser = argparse.ArgumentParser(description='Evaluate fault localization results.')
    arg_parser.add_argument("-r", "--result_file", required=False, type=str, default=DEFAULT_INPUT,
                            help="The file conataining test results")

    args = arg_parser.parse_args()

    with gzip.open(args.result_file, "rb") as f:
        meta_evaluation = pickle.load(f)

    print(meta_evaluation.event_processor.translators)
    combining_method = CombineMaxThenAvg()
    evaluation = meta_evaluation.evaluate(OchiaiCoefficient, combining_method)

    print(evaluation.fraction_top_k_accurate)
    print(evaluation.avg_recall_at_k)
    print(evaluation.avg_precision_at_k)
