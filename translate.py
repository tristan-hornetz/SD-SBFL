import argparse
import os
import sys
import gzip
import pickle

from Evaluator.EventTranslation import EventProcessor, DEFAULT_TRANSLATORS
from Evaluator.Evaluation import MetaEvaluation

if __name__ == "__main__":
    DEFAULT_INPUT = os.path.dirname(os.path.abspath(sys.argv[0])) + "/thefuck"
    DEFAULT_OUTPUT = ""

    arg_parser = argparse.ArgumentParser(description='Translate raw, recorded events to Evaluation Framework events.')
    arg_parser.add_argument("-d", "--directory", required=False, type=str, default=DEFAULT_INPUT,
                            help="The file containing the recorded events")
    arg_parser.add_argument("-o", "--output_file", required=False, type=str, default=DEFAULT_OUTPUT,
                            help="The output file")

    args = arg_parser.parse_args()
    meta_evaluation = MetaEvaluation(EventProcessor(DEFAULT_TRANSLATORS))
    meta_evaluation.add_from_directory(args.directory)

    output_file = args.output_file

    if output_file == "":
        _results = meta_evaluation.meta_rankings[0]._results
        output_file = f"./translated_results_{_results.project_name}.pickle.gz"

    if os.path.exists(output_file):
        os.remove(output_file)

    with gzip.open(output_file, "xb") as f:
        pickle.dump(meta_evaluation, f)












