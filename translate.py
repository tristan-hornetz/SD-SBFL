import argparse
import gzip
import os
import pickle
import sys

from Evaluator.CodeInspection.utils import mkdirRecursive
from Evaluator.EventTranslation import EventProcessor, DEFAULT_TRANSLATORS
from Evaluator.Ranking import MetaRanking


def translate_file(path, event_processor, output_dir):
    assert os.path.exists(path)
    try:
        with gzip.open(path) as f:
            _results = pickle.load(f)
        mr = MetaRanking(*event_processor.process(_results), _results)
        output_file = f"{output_dir}/translated_results_{_results.project_name}_{_results.bug_id}.pickle.gz"
        if os.path.exists(output_file):
            os.remove(output_file)
        with gzip.open(output_file, "xb") as f:
            pickle.dump(mr, f)
        print("Succeeded " + path)
    except:
        print("Failed " + path)


def translate_directory(path, event_processor, output_dir):
    assert os.path.isdir(path)
    if not os.path.exists(os.path.abspath(output_dir)):
        mkdirRecursive(output_dir)
    for filename in sorted(os.listdir(path)):
        translate_file(f"{str(path)}/{filename}", event_processor, output_dir)


if __name__ == "__main__":
    DEFAULT_OUTPUT = ""

    arg_parser = argparse.ArgumentParser(description='Translate raw, recorded events to Evaluation Framework events.')
    arg_parser.add_argument("-d", "--directory", required=True, type=str, default="",
                            help="The file containing the recorded events")
    arg_parser.add_argument("-o", "--output_dir", required=False, type=str, default=DEFAULT_OUTPUT,
                            help="The output diretory")

    args = arg_parser.parse_args()
    event_processor = EventProcessor(DEFAULT_TRANSLATORS)
    output_dir = args.output_dir
    if output_dir == "":
        output_dir = os.path.dirname(
            os.path.abspath(sys.argv[0])) + f"/results_translated/{os.path.basename(os.path.abspath(args.directory))}"
    translate_directory(args.directory, event_processor, output_dir)
