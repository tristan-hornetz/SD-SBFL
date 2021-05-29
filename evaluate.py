import argparse
import os
import sys
import gzip
import pickle

MATCH_STRING = "tornado/http1connection.py"

DEFAULT_INPUT = os.path.dirname(os.path.abspath(sys.argv[0])) + "/TestWrapper/results.pickle.gz"
DEFAULT_INFODIR = "."

arg_parser = argparse.ArgumentParser(description='Evaluate fault localization results.')
arg_parser.add_argument("-r", "--result_file", required=False, type=str, default=DEFAULT_INPUT, help="The file conataining test results")
arg_parser.add_argument("-i", "--info_dir", required=False, type=str, default=DEFAULT_INFODIR, help="The BugsInPy Project Directory containig information about the specific bug")

args = arg_parser.parse_args()


with gzip.open(args.result_file) as f:
    result_container = pickle.load(f)

results = result_container.results
bug_id = result_container.bug_id
project_name = result_container.project_name


print(f"Results for {project_name}, Bug {bug_id}")
print(f"Ranked {len(results)} Events")
print(f"Most suspicious:\n{results[0]}\n")

matchstring_index = -1
matchstring_item = ("", 0)
for item, lineno in results:
    matchstring_index += 1
    if MATCH_STRING in item:
        matchstring_item = (item, lineno)
        break

print(f"Most suspicious occurrence of matchstring module: Rank #{matchstring_index}, Top {matchstring_index*100.0/len(results)}%")
print(matchstring_item)

