import argparse
import os
import gzip
import pickle
import sys

from run_single_test import run_test

FIX_REPORT_FILENAME = "results_fix.txt"

def get_files_recursively(directory: str):
    filenames = []
    for filename in os.listdir(directory):
        if os.path.isdir(f"{directory}/{filename}"):
            filenames.extend(get_files_recursively(f"{directory}/{filename}"))
        else:
            filenames.append(f"{directory}/{filename}")
    return filenames


def validate(filename):
    try:
        with gzip.open(filename, "rb") as f:
            _results = pickle.load(f)
    except:
        print(f"{filename} - Invalid, not result file")
        return False, None
    try:
        assert len(_results.results) > 0
        assert len(_results.collectors_with_event[_results.FAIL]) > 0
        print(f"{filename} - Valid")
        return True, _results
    except AssertionError:
        print(f"{filename} - Invalid, assertion error")
    except AttributeError:
        print(f"{filename} - Invalid, attribute error")
    return False, _results


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Check whether results from Recorder are valid for evaluation')
    arg_parser.add_argument("-d", "--directory", required=True, type=str, default=None,
                            help="The file containing the recorded events")
    arg_parser.add_argument("-f", "--fix", help="Attempt to fix the invalid result files", action='store_true')
    arg_parser.add_argument("-r", "--recursive", help="Search given directory recursively", action='store_true')

    args = arg_parser.parse_args()

    result_files = 0
    valid_result_files = 0
    invalid_files = []
    for filename in sorted(get_files_recursively(args.directory)) if args.recursive \
            else sorted(map(lambda s: f"{args.directory}/{s}", os.listdir(args.directory))):
        valid, _results = validate(filename)
        if _results is not None:
            result_files += 1
            if valid:
                valid_result_files += 1
            else:
                invalid_files.append((filename, _results.project_name, _results.bug_id))

    print(f"\nResult files: {result_files}\nValid result files: {valid_result_files}\nValid percentage: {valid_result_files*100.0/result_files}%")

    if args.fix:
        fixed = []
        not_fixed = []
        for f, project_name, bug_id in invalid_files:
            root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            run_test(root_dir, project_name, bug_id, os.path.abspath(f))
            valid, _results = validate(f)
            if valid and _results is not None:
                fixed.append(f)
            else:
                not_fixed.append(f)

        report_str = f"FIXED: {len(fixed)}\nNOT FIXED: {len(not_fixed)}\nTOTAL: {len(invalid_files)}\n\n"
        report_str += f"FIXED FILES:\n{(os.linesep + '    ').join(fixed)}\n\n"
        report_str += f"UNFIXED FILES:\n{(os.linesep + '    ').join(not_fixed)}"
        if os.path.exists(FIX_REPORT_FILENAME):
            os.remove(FIX_REPORT_FILENAME)
        with open(FIX_REPORT_FILENAME, "xt") as f:
            f.write(report_str)


