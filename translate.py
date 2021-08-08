import argparse
import gzip
import os
import pickle
import sys
import time

from multiprocessing import Process

from Evaluator.CodeInspection.utils import mkdirRecursive
from Evaluator.EventTranslation import EventProcessor, DEFAULT_TRANSLATORS
from Evaluator.Ranking import MetaRanking


def translate_file(path, event_processor, output_dir):
    assert os.path.exists(path)
    try:
        with gzip.open(path) as f:
            _results = pickle.load(f)
        print(f"Translating results from {_results.project_name}, bug {_results.bug_id}")
        mr = MetaRanking(*event_processor.process(_results), _results)
        output_file = f"{output_dir}/{_results.project_name}/translated_results_{_results.project_name}_{_results.bug_id}.pickle.gz"
        if not os.path.exists(os.path.dirname(os.path.abspath(output_file))):
            mkdirRecursive(os.path.dirname(os.path.abspath(output_file)))
        if os.path.exists(output_file):
            os.remove(output_file)
        with gzip.open(output_file, "xb") as f:
            pickle.dump(mr, f)
        print("Succeeded " + path)
    except:
        print("Failed " + path)


def translate_directory(path, event_processor, output_dir):
    assert os.path.isdir(path)
    for filename in sorted(os.listdir(path)):
        if os.path.isdir(f"{str(path)}/{filename}"):
            continue
        translate_file(f"{str(path)}/{filename}", event_processor, output_dir)
    return


def get_subdirs_recursive(start_path):
    contents = os.listdir(start_path)
    dirs = []
    for f in contents:
        if os.path.isdir(f"{start_path}/{f}"):
            dirs.append(os.path.realpath(f"{start_path}/{f}"))
            dirs.extend(get_subdirs_recursive(os.path.realpath(f"{start_path}/{f}")))
    return dirs


def translate_directory_parallel(path, event_processor, output_dir, threads=-1):
    if threads < 1:
        threads = os.cpu_count()
    dirs = get_subdirs_recursive(path) + [path]
    processes = [Process(target=translate_directory, args=(d, event_processor, output_dir), name=d) for d in dirs]
    active_processes = []
    while len(processes) > 0:
        while len(active_processes) < threads and len(processes) > 0:
            p = processes.pop()
            p.start()
            active_processes.append(p)
        for p in active_processes.copy():
            if not p.is_alive():
                active_processes.remove(p)
        time.sleep(.1)


if __name__ == "__main__":
    DEFAULT_OUTPUT = ""

    arg_parser = argparse.ArgumentParser(description='Translate raw, recorded events to Evaluation Framework events.')
    arg_parser.add_argument("-d", "--directory", required=True, type=str, default="",
                            help="The file containing the recorded events")
    arg_parser.add_argument("-r", "--recursive", help="Search for files recursively", action='store_true')
    arg_parser.add_argument("-o", "--output_dir", required=False, type=str, default=DEFAULT_OUTPUT,
                            help="The output diretory")

    args = arg_parser.parse_args()
    event_processor = EventProcessor(DEFAULT_TRANSLATORS)
    output_dir = args.output_dir
    if output_dir == "":
        output_dir = os.path.dirname(
            os.path.abspath(sys.argv[0])) + f"/results_translated"
    if args.recursive:
        translate_directory_parallel(args.directory, event_processor, output_dir)
    else:
        translate_directory(args.directory, event_processor, output_dir)
