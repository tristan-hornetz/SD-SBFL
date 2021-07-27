import argparse
import os
import gzip
import pickle
import sys

from run_single_test import run_test

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Check whether results from Recorder are valid for evaluation')
    arg_parser.add_argument("-d", "--directory", required=True, type=str, default=None,
                            help="The file containing the recorded events")
    arg_parser.add_argument("-f", "--fix", help="Attempt to fix the invalid result files", action='store_true')

    args = arg_parser.parse_args()

    result_files = 0
    valid_result_files = 0
    invalid_files = []
    for filename in sorted(os.listdir(args.directory)):
        try:
            with gzip.open(f"{str(args.directory)}/{filename}", "rb") as f:
                _results = pickle.load(f)
            result_files += 1
        except:
            print(f"{str(args.directory)}/{filename} - Invalid, not result file")
            continue
        try:
            assert len(_results.results) > 0
            assert len(_results.collectors_with_event[_results.FAIL]) > 0
            print(f"{str(args.directory)}/{filename} - Valid")
            valid_result_files += 1
        except AssertionError:
            print(f"{str(args.directory)}/{filename} - Invalid, assertion error")
            invalid_files.append((f"{str(args.directory)}/{filename}", _results))
        except AttributeError:
            print(f"{str(args.directory)}/{filename} - Invalid, attribute error")
            invalid_files.append((f"{str(args.directory)}/{filename}", _results))

    print(f"\nResult files: {result_files}\nValid result files: {valid_result_files}\nValid percentage: {valid_result_files*100.0/result_files}%")

    if args.fix:
        for f, r in invalid_files:
            project_name = r.project_name
            bug_id = r.bug_id
            root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            run_test(root_dir, project_name, bug_id, os.path.abspath(f))

