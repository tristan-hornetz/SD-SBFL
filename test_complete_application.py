import os
import sys

from Evaluator.CodeInspection.utils import mkdirRecursive
from run_single_test import run_test

if __name__ == "__main__":
    import argparse

    root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    arg_parser = argparse.ArgumentParser(
        description="Run all tests for a specific application"
    )
    arg_parser.add_argument(
        "-p",
        "--project_name",
        required=True,
        type=str,
        default="thefuck",
        help="The name of the target project",
    )
    arg_parser.add_argument(
        "-o",
        "--output_directory",
        required=False,
        type=str,
        default=root_dir + "/results",
        help="The directory in which the output should be stored",
    )
    arg_parser.add_argument(
        "-s",
        "--start_at",
        required=False,
        type=int,
        default=1,
        help="Start testing at the bug with the given id (default=1)",
    )

    args = arg_parser.parse_args()
    root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    project = args.project_name

    bugsinpy_directory = root_dir + "/_BugsInPy/projects/" + project + "/bugs"
    if not os.path.isdir(bugsinpy_directory):
        print(f"The project '{project}' is not part of BugsInPy.")
        exit(-1)

    dump_dir = args.output_directory + "/" + project
    if not os.path.isdir(dump_dir):
        mkdirRecursive(dump_dir)

    bugs = list(
        int(b)
        for b in (
            filter(
                lambda n: str(n).lstrip("0").isnumeric(), os.listdir(bugsinpy_directory)
            )
        )
    )
    bugs.sort()
    for bug in bugs[args.start_at - 1 :]:
        run_test(
            root_dir,
            project,
            int(bug),
            output_file=dump_dir + f"/{project}_{bug}.pickle.gz",
        )
