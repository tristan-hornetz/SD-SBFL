import gzip
import pickle

from os.path import dirname, exists, abspath

from Evaluator.CodeInspection.utils import mkdirRecursive
from Evaluator.CombiningMethod import *
from Evaluator.Evaluation import Evaluation

THREADS = os.cpu_count()


def create_evaluation(
    result_dir,
    similarity_coefficient,
    combining_method: CombiningMethod,
    save_destination="",
    print_results=False,
    num_threads=-1,
):

    evaluation = Evaluation(similarity_coefficient, combining_method)
    evaluation.add_directory(result_dir, THREADS if num_threads < 1 else num_threads)

    if save_destination != "":
        if not exists(dirname(save_destination)):
            mkdirRecursive(abspath(dirname(save_destination)))
        if exists(save_destination):
            os.remove(save_destination)
        with gzip.open(save_destination, "xb") as f:
            pickle.dump(evaluation, f)

    if print_results:
        print(len(evaluation.rankings))
        print("\n")

        print(evaluation.fraction_top_k_accurate)
        print(evaluation.avg_recall_at_k)
        print(evaluation.avg_precision_at_k)
