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
from Evaluator.Evaluation import MetaEvaluation

if __name__ == "__main__":
    DEFAULT_INPUT = os.path.dirname(os.path.abspath(sys.argv[0])) + "/thefuck"
    DEFAULT_OUTPUT = os.path.dirname(os.path.abspath(sys.argv[0])) + "/translated_results.pickle.gz"

    arg_parser = argparse.ArgumentParser(description='Translate raw, recorded events to Evaluation Framework events.')
    arg_parser.add_argument("-d", "--directory", required=False, type=str, default=DEFAULT_INPUT,
                            help="The file containing the recorded events")
    arg_parser.add_argument("-o", "--output_file", required=False, type=str, default=DEFAULT_OUTPUT,
                            help="The output file")

    args = arg_parser.parse_args()
    meta_evaluation = MetaEvaluation(EventProcessor(DEFAULT_TRANSLATORS))
    meta_evaluation.add_from_directory(args.directory)

    evaluation = meta_evaluation.evaluate(OchiaiCoefficient, CombineMaxThenAvg())










