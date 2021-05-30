import argparse
import gzip
import os
import pickle
import sys

from unidiff import PatchSet


def get_info_directory(_results):
    base = _results.work_dir.split("_BugsInPy")[0]
    return f"{base}_BugsInPy/projects/{_results.project_name}/bugs/{_results.bug_id}"


class BugInfo:
    def __init__(self, _results):
        self.info_dir = get_info_directory(_results)
        with open(self.info_dir + "/bug.info", "rt") as f:
            self.attrs = list()
            while True:
                line = f.readline()
                if not line:
                    break
                if "=" in line:
                    attr = line.split("=", 1)[0]
                    val = line.split("=", 1)[1].strip("\" \n")
                    setattr(self, attr, val)
                    self.attrs.append(attr)

    def __str__(self):
        ret = ""
        for a in self.attrs:
            ret += f"{a}=\"{getattr(self, a)}\"\n"
        return ret


def getPatchSet(info: BugInfo):
    with open(info.info_dir + "/bug_patch.txt", "rt") as f:
        diff_text = f.read()
    return PatchSet(diff_text)


def getMatchString(info: BugInfo):
    patch_set = getPatchSet(info)
    file = patch_set.modified_files[0]
    return file.path + "["


class SFL_Evaluation:
    def __init__(self, result_file):
        with gzip.open(result_file) as f:
            self.result_container = pickle.load(f)

        self.match_string = getMatchString(BugInfo(self.result_container))
        self.matchstring_index = -1
        self.matchstring_item = ("", 0)
        for item, lineno in self.result_container.results:
            self.matchstring_index += 1
            if self.match_string in item:
                self.matchstring_item = (item, lineno)
                break

    def __str__(self):
        return f"Results for {self.result_container.project_name}, Bug {self.result_container.bug_id}\n" + \
               f"Ranked {len(self.result_container.results)} Events\n" + \
               f"Most suspicious:\n{self.result_container.results[0]}\n\n" + \
               f"Most suspicious occurrence of buggy module: Rank #{self.matchstring_index + 1}, " + \
               f"Top {(self.matchstring_index + 1) * 100.0 / len(self.result_container.results)}%\n" + \
               f"{self.matchstring_item}"


if __name__ == "__main__":
    DEFAULT_INPUT = os.path.dirname(os.path.abspath(sys.argv[0])) + "/TestWrapper/results.pickle.gz"

    arg_parser = argparse.ArgumentParser(description='Evaluate fault localization results.')
    arg_parser.add_argument("-r", "--result_file", required=False, type=str, default=DEFAULT_INPUT,
                            help="The file conataining test results")

    args = arg_parser.parse_args()
    evaluation = SFL_Evaluation(args.result_file)
    print(evaluation)
    print("\n")
    for r in evaluation.result_container.results[:10]:
        print(r)

