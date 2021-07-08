import argparse
import os
import random
import sys
import gzip
import pickle

from Evaluator.EventTranslation import EventProcessor, DEFAULT_TRANSLATORS
from Evaluator.Ranking import Ranking
from Evaluator.SimilarityCoefficient import OchiaiCoefficient
from Evaluator.CombiningMethod import CombineMaxThenAvg
from Evaluator.CodeInspection.Methods import getBuggyMethods

if __name__ == "__main__":
    DEFAULT_INPUT = os.path.dirname(os.path.abspath(sys.argv[0])) + "/TestWrapper/results.pickle.gz"
    DEFAULT_OUTPUT = os.path.dirname(os.path.abspath(sys.argv[0])) + "/translated_results.pickle.gz"

    arg_parser = argparse.ArgumentParser(description='Translate raw, recorded events to Evaluation Framework events.')
    arg_parser.add_argument("-r", "--result_file", required=False, type=str, default=DEFAULT_INPUT,
                            help="The file containing the recorded events")
    arg_parser.add_argument("-o", "--output_file", required=False, type=str, default=DEFAULT_INPUT,
                            help="The output file")

    args = arg_parser.parse_args()
    with gzip.open(args.result_file) as f:
        _results = pickle.load(f)

    processor = EventProcessor(DEFAULT_TRANSLATORS)

    events, methods, info = processor.process(_results)

    print(len(_results.results))
    print(len(events.events))
    print(len(methods.keys()))

    ranking = Ranking(events, methods, OchiaiCoefficient, CombineMaxThenAvg())

    print("\n".join(str(e[0]) + " - " + str(e[1]) for e in ranking.ranking[:10]))

    print("\n" * 2)

    print("\n".join(str(e) for e in getBuggyMethods(_results, info)))





