import os
import argparse
import sys

from TestWrapper.root.Evaluation import SFL_Evaluation
from TestWrapper.root.Predicates import NoPredicate

if __name__ == "__main__":
    DEFAULT_INPUT = os.path.dirname(os.path.abspath(sys.argv[0])) + "/TestWrapper/results.pickle.gz"

    arg_parser = argparse.ArgumentParser(description='Evaluate fault localization results.')
    arg_parser.add_argument("-r", "--result_file", required=False, type=str, default=DEFAULT_INPUT,
                            help="The file conataining test results")

    args = arg_parser.parse_args()
    evaluation = SFL_Evaluation(args.result_file)

    result_dump = os.path.dirname(os.path.abspath(
        sys.argv[0])) + f"/results_{evaluation.result_container.project_name}_{evaluation.result_container.bug_id}.txt"
    with open(result_dump, "wt") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        print(evaluation)
        print("\n\nTop 10 most suspicious methods:\n")
        i = 0
        for r in evaluation.result_methods[:10]:
            i += 1
            print(f"#{i}: {str(r)}")
        print("\n\nTop 10 most suspicious raw events:\n")
        i = 0
        ranker = evaluation.ranker_type(evaluation.result_container, evaluation.bug_info, predicates=[NoPredicate(evaluation.result_container)])
        for r, s in ranker.rank()[:10]:
            i += 1
            print(f"#{i}: {r} - Suspiciousness {s}")

        sys.stdout = old_stdout
    os.system(f"less \"{result_dump}\"")  #
    print("Results have been written to " + result_dump)
