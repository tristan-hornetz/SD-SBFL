import argparse
import gzip
import inspect
import os
import pickle
import sys
import ast
import pkg_resources


installed = {pkg.key for pkg in pkg_resources.working_set}
if 'gitpython' in installed and 'unidiff' in installed:
    from git import Repo
    from unidiff import PatchSet, PatchedFile
else:
    Repo = PatchSet = PatchedFile = object



def get_info_directory(_results):
    base = _results.work_dir.split("_BugsInPy")[0]
    return f"{base}_BugsInPy/projects/{_results.project_name}/bugs/{_results.bug_id}"


class BugInfo:
    def parse_file(self, file_name, attr_prefix=""):
        with open(file_name, "rt") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if "=" in line:
                    attr = attr_prefix + line.split("=", 1)[0]
                    val = line.split("=", 1)[1].strip("\" \n")
                    setattr(self, attr, val)
                    self.attrs.append(attr)

    def __init__(self, _results):
        self.info_dir = get_info_directory(_results)
        self.attrs = list()
        self.parse_file(self.info_dir + "/bug.info")
        self.parse_file(self.info_dir + "/../../project.info", "project_")

    def __str__(self):
        ret = ""
        for a in self.attrs:
            ret += f"{a}=\"{getattr(self, a)}\"\n"
        return ret


def getPatchSet(info: BugInfo):
    with open(info.info_dir + "/bug_patch.txt", "rt") as f:
        diff_text = f.read()
    return PatchSet(diff_text)


def getNodeParents(node: ast.AST, parent_dict: dict, parents: list, line_nodes: list, lineno):
    parent_dict[node] = parents
    if (node.lineno if hasattr(node, 'lineno') else -1) == lineno:
        line_nodes.append(node)
    new_parents = parents.copy() + [node]
    for child in ast.iter_child_nodes(node):
        getNodeParents(child, parent_dict, new_parents, line_nodes, lineno)
    return parent_dict, line_nodes


def getParentFunctionFromLineNo(source: ast.AST, lineno: int):
    parent_dict, line_nodes = getNodeParents(source, dict(), list(), list(), lineno)

    if len(line_nodes) == 0:
        return None

    parent_functions = filter(lambda node: isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef),
                              parent_dict[line_nodes[0]])

    parents = list(reversed(list(parent_functions)))
    return parents[0].name if len(parents) > 0 else None


def getBuggyMethodsFromFile(_results, project_root: str, file: PatchedFile, is_target_file: bool):
    linenos = set()
    for hunk in file:
        start = hunk.source_start if not is_target_file else hunk.target_start
        _r = list(range(start, start + (hunk.source_length if not is_target_file else hunk.target_length)))
        linenos.update(_r)

    abs_file = project_root + "/" + file.path
    with open(abs_file, "rt") as f:
        source = ast.parse(f.read(), filename=abs_file)

    return set(filter(lambda o: o is not None, (getParentFunctionFromLineNo(source, lineno) for lineno in linenos)))


def getRepoObj(_results, info: BugInfo, dir_name="", fixed=False):
    if dir_name == "":
        dir_name = _results.work_dir + "/" + _results.project_name
    try:
        if os.path.isdir(dir_name):
            repo = Repo(dir_name)
            repo.git.checkout(info.fixed_commit_id if fixed else info.buggy_commit_id)
            return repo
    except Exception:
        pass
    return None


def getValidProjectDir(_results, info, fixed=False):
    directory = os.path.dirname(inspect.getfile(getNodeParents)) + "/.temp"
    repo = getRepoObj(_results, info, directory, fixed)
    if repo is not None:
        return _results.work_dir + "/" + _results.project_name
    if os.path.exists(directory):
        os.system(f"rm -rf \"{directory}\"")
    os.mkdir(directory)
    repo = Repo.clone_from(info.project_github_url, to_path=directory)
    repo.git.checkout(info.buggy_commit_id if not fixed else info.fixed_commit_id)
    return directory


def getBuggyMethods(_results, info: BugInfo):
    patch_set = getPatchSet(info)
    methods = set()
    repo_dir = getValidProjectDir(_results, info, False)
    for file in patch_set.modified_files:
        methods.update((file.path, method) for method in getBuggyMethodsFromFile(_results, repo_dir, file, False))
    repo_dir = getValidProjectDir(_results, info, True)
    for file in patch_set.modified_files:
        methods.update((file.path, method) for method in getBuggyMethodsFromFile(_results, repo_dir, file, True))
    return list(methods)


class SFL_Evaluation:
    def __init__(self, result_file):
        with gzip.open(result_file) as f:
            self.result_container = pickle.load(f)

        self.bug_info = BugInfo(self.result_container)
        self.buggy_methods = getBuggyMethods(self.result_container, self.bug_info)

        self.matchstring_index = len(self.result_container.results)
        self.matchstring_item = ("", 0)
        for b_file, b_method in self.buggy_methods:
            matchstring_index = -1
            match_string = f"{b_file}[{b_method}]"
            for item, lineno in self.result_container.results:
                matchstring_index += 1
                if match_string in item and matchstring_index < self.matchstring_index:
                    self.matchstring_item = (item, lineno)
                    self.matchstring_index = matchstring_index
                    break

    def __str__(self):
        return f"Results for {self.result_container.project_name}, Bug {self.result_container.bug_id}\n" + \
               f"Ranked {len(self.result_container.results)} Events\n\n" + \
               f"There {'is one buggy function' if len(self.buggy_methods) == 1 else f'are {len(self.buggy_methods)} buggy functions'} in this commit: \n" + \
               f"{os.linesep.join(list(f'    {filename}: {method}'  for filename, method in self.buggy_methods))}\n\n" + \
               f"Most suspicious event:\n{self.result_container.results[0]}\n\n" + \
               f"Most suspicious event in a buggy function: Rank #{self.matchstring_index + 1}, " + \
               f"Top {(self.matchstring_index + 1) * 100.0 / len(self.result_container.results)}%\n" + \
               f"{self.matchstring_item}"


if __name__ == "__main__":
    DEFAULT_INPUT = os.path.dirname(os.path.abspath(sys.argv[0])) + "/TestWrapper/results.pickle.gz"

    arg_parser = argparse.ArgumentParser(description='Evaluate fault localization results.')
    arg_parser.add_argument("-r", "--result_file", required=False, type=str, default=DEFAULT_INPUT,
                            help="The file conataining test results")

    args = arg_parser.parse_args()
    evaluation = SFL_Evaluation(args.result_file)

    result_dump = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/results_{evaluation.result_container.project_name}_{evaluation.result_container.bug_id}.txt"
    with open(result_dump, "wt") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        print(evaluation)
        print("\n\nTop 10 most suspicious events:\n")
        i = 0
        for r in evaluation.result_container.results[:10]:
            i += 1
            print(f"#{i}: {r}")
        sys.stdout = old_stdout
    os.system(f"less \"{result_dump}\"")#
    print("Results have been written to " + result_dump)
