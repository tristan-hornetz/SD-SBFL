import argparse
import ast
import gzip
import inspect
import os
import pickle
import sys

import pkg_resources

installed = {pkg.key for pkg in pkg_resources.working_set}
if 'gitpython' in installed and 'unidiff' in installed:
    from git import Repo
    from unidiff import PatchSet, PatchedFile
else:
    Repo = PatchSet = PatchedFile = object


def get_info_directory(_results):
    """
    Get the absolute path of the BugsInPy directory containing information on the bug tested in _results
    :param _results: The SFL_Results of a test run
    :return: The absolute path of the BugsInPy directory containing information on the bug tested in _results
    """
    base = _results.work_dir.split("_BugsInPy")[0]
    return f"{base}_BugsInPy/projects/{_results.project_name}/bugs/{_results.bug_id}"


class BugInfo:
    """
    Container class for contents of bug.info and project.info
    """

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
        """
        Create a BugInfo instance from the SFL_Results of a test run
        :param _results: The SFL_Results of a test run
        """
        self.info_dir = get_info_directory(_results)
        self.attrs = list()
        self.parse_file(self.info_dir + "/bug.info")
        self.parse_file(self.info_dir + "/../../project.info", "project_")

    def __str__(self):
        ret = ""
        for a in self.attrs:
            ret += f"{a}=\"{getattr(self, a)}\"\n"
        return ret


class LineNumberExtractor(ast.NodeVisitor):
    def __init__(self, *args, **kwargs):
        super(LineNumberExtractor, self).__init__(*args, **kwargs)
        self.linenos = set()

    def visit(self, node: ast.AST):
        if hasattr(node, 'lineno'):
            self.linenos.add(node.lineno)
        return super().visit(node)


class DebuggerMethod:
    """
    Represents a Method, as extracted from code
    """
    def __init__(self, name: str, file: str, linenos=None):
        if linenos is None:
            linenos = set()
        self.name = name
        self.file = file
        self.linenos = linenos

    def add_lineno(self, lineno):
        self.linenos.add(lineno)

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return f"{self.file}[{self.name} | Lines {';'.join(str(n) for n in sorted(self.linenos))}]"


def getPatchSet(info: BugInfo):
    """
    Create a unidiff PatchSet instance from the bug_patch.txt file of a specific Bug
    :param info: A BugInfo instance for the bug to be loaded
    :return: A unidiff PatchSet instance from the bug_patch.txt file of the bug from info
    """
    with open(info.info_dir + "/bug_patch.txt", "rt") as f:
        diff_text = f.read()
    return PatchSet(diff_text)


def getNodeParents(node: ast.AST, parent_dict: dict, parents: list, line_nodes: list, lineno: int):
    """
    Recursive function walking the subnodes of node to produce a list of nodes with a certain lineno. Also
    returns a dictionary containing the parent nodes for each node
    :param node: The ASR to be walked on
    :param parent_dict: Should be an empty dictionary
    :param parents: Should be an empty list
    :param line_nodes: Should be an empty list
    :param lineno: The lineno of nodes to be collected
    :return: parent_dict, line_nodes
    """
    parent_dict[node] = parents
    if (node.lineno if hasattr(node, 'lineno') else -1) == lineno:
        line_nodes.append(node)
    new_parents = parents.copy() + [node]
    for child in ast.iter_child_nodes(node):
        getNodeParents(child, parent_dict, new_parents, line_nodes, lineno)
    return parent_dict, line_nodes


def getParentFunctionFromLineNo(source: ast.AST, lineno: int):
    """
    Get the name of the function containing a specific line of code
    :param source: The AST to be processed
    :param lineno: The lineno to get the function of
    :return: The name of the function, or None if no function could be found, linenos of the first parent
    """
    parent_dict, line_nodes = getNodeParents(source, dict(), list(), list(), lineno)
    if len(line_nodes) == 0:
        return None

    parent_elements = filter(lambda node: isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef)
                                          or isinstance(node, ast.AsyncFunctionDef),
                              parent_dict[line_nodes[0]])

    parents = list(parent_elements)
    if len(parents) == 0:
        return None
    extractor = LineNumberExtractor()
    extractor.visit(next(reversed(parents)))
    return "::".join(p.name for p in parents), tuple(sorted(extractor.linenos))


def getBuggyMethodsFromFile(project_root: str, file: PatchedFile, is_target_file: bool):
    """
    Get a set of modified methods from a unidiff PatchedFile object
    :param project_root: The root directory of a Git repository containing the modified file
    :param file: The PatchedFile object
    :param is_target_file: Is the repository in a fixed state?
    :return: A set of modified methods
    """
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
    """
    Create a GitPython Repo object for a directory and checkout the buggy (or fixed) commit
    :param _results: The SFL_Results of the test run
    :param info: The BugInfo for _results
    :param dir_name: The directory containing the git repo
    :param fixed: Checkout the fixed commit?
    :return: A GitPython Repo object for dir_name
    """
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


def getValidProjectDir(_results, info, fixed=False, directory="", instrument=False):
    """
    Create a valid Git Repo for a specific bug and return the path to it
    :param _results: The SFL_Results of the test run
    :param info: The BugInfo for _result
    :param fixed: Checkout the fixed commit?
    :param directory: The directory to create the repo in
    :return: The path to a valid git repo for the bug in info
    """
    if directory == "":
        directory = os.path.dirname(inspect.getfile(getNodeParents)) + "/.temp"
    if os.path.exists(directory):
        os.system(f"rm -rf \"{directory}\"")
    os.mkdir(directory)
    #repo = Repo.clone_from(info.project_github_url, to_path=directory)
    #repo.git.checkout(info.buggy_commit_id if not fixed else info.fixed_commit_id)
    root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    binary_dir = root_dir + '/_BugsInPy/framework/bin'
    os.system(f'{binary_dir}/bugsinpy-checkout -p {_results.project_name} -i {_results.bug_id} -v {1 if fixed else 0} -w {directory}')
    if instrument:
        debugger_module = os.path.abspath(root_dir + '/run_test.py')
        os.system(f'{binary_dir}/bugsinpy-instrument -f -c {debugger_module} -w {directory + "/" + _results.project_name}')

    return directory + "/" + _results.project_name


def getBuggyMethods(_results, info: BugInfo):
    """
    Get a list of all methods that are were modified to fix a specific bug
    :param _results: The SFL_Results of the test run
    :param info: The BugInfo for _result
    :return: A list of all methods that are were modified to fix a specific bug
    """
    patch_set = getPatchSet(info)
    methods = set()
    repo_dir = getValidProjectDir(_results, info, False, instrument=True)
    for file in patch_set.modified_files:
        methods.update((file.path, method, linenos) for method, linenos in getBuggyMethodsFromFile(repo_dir, file, False))
    repo_dir = getValidProjectDir(_results, info, True, instrument=True)
    for file in patch_set.modified_files:
        methods.update((file.path, method, linenos) for method, linenos in getBuggyMethodsFromFile(repo_dir, file, True))
    return list(methods)


def extractMethodsFromFile(directory, file, methods):
    if os.path.exists(directory + "/" + file):
        with open(directory + "/" + file, "rt") as f:
            node = ast.parse(f.read())
        function_defs = set(filter(lambda n: isinstance(n, ast.FunctionDef) or isinstance(n, ast.AsyncFunctionDef), ast.walk(node)))
        lines_per_method = list()
        for d in filter(lambda n: n.name in methods, function_defs):
            extractor = LineNumberExtractor()
            extractor.visit(d)
            lines_per_method.append((d.name, extractor.linenos))
        return lines_per_method
    return list()


def extractMethodsFromCode(_results, info, method_strings):
    methods_per_file = dict()
    for s in method_strings:
        m_arr = s.split("[")
        if len(m_arr) > 1:
            if m_arr[0] in methods_per_file.keys():
                methods_per_file[m_arr[0]].add(m_arr[1].split("]")[0])
            else:
                methods_per_file[m_arr[0]] = {m_arr[1].split("]")[0]}
    directory = getValidProjectDir(_results, info, False, instrument=True)
    method_objects = set()
    for file in methods_per_file.keys():
        lines_per_method = extractMethodsFromFile(directory, file, methods_per_file[file])
        for method, linenos in lines_per_method:
            method_objects.add(DebuggerMethod(method, file, linenos))
    return list(method_objects)

class SFL_Evaluation:
    """
    Container class for all information that is relevant for a test run
    """

    def sortResultMethods(self, work_dir):
        method_dict = dict()
        ranks = dict()
        for method in self.result_methods:
            ranks[method] = len(self.result_container.results)
            if method.file in method_dict.keys():
                if method.name in method_dict[method.file].keys():
                    method_dict[method.file][method.name].append((method.linenos, method))
                else:
                    method_dict[method.file][method.name] = [(method.linenos, method)]
            else:
                method_dict[method.file] = {method.name: [(method.linenos, method)]}
        index = -1
        for event, lineno in self.result_container.results:
            index += 1
            method_str = self.getMethodStringFromEvent(event, work_dir)
            m_arr = method_str.split("[", 1)
            if m_arr[0] in method_dict.keys():
                m_name = m_arr[1].split("]")[0]
                if m_name in method_dict[m_arr[0]].keys():
                    for linenos, method in method_dict[m_arr[0]][m_name]:
                        if lineno in linenos:
                            ranks[method] = min(index, ranks[method])
        self.result_methods.sort(key=lambda m: ranks[m])

    def getMethodStringFromEvent(self, event: str, work_dir):
        return next(reversed(event.split(" @ "))).replace(work_dir + "/", "")

    def __init__(self, result_file):
        with gzip.open(result_file) as f:
            self.result_container = pickle.load(f)

        self.result_method_strings = set()

        self.result_methods = list()

        work_dir = self.result_container.work_dir + "/" + self.result_container.project_name
        for event, lineno in self.result_container.results:
            method = self.getMethodStringFromEvent(event, work_dir)
            self.result_method_strings.add(method)

        self.bug_info = BugInfo(self.result_container)
        self.buggy_methods = getBuggyMethods(self.result_container, self.bug_info)
        self.result_methods = extractMethodsFromCode(self.result_container, self.bug_info, self.result_method_strings)
        self.sortResultMethods(work_dir)

        self.highest_rank = len(self.result_methods)

        for b_file, b_method, b_linenos in self.buggy_methods:
            current_index = -1
            for method in self.result_methods:
                current_index += 1
                if self.highest_rank <= current_index:
                    break
                if method.file in b_file and method.name == b_method:
                    if len(set(b_linenos).intersection(method.linenos)) > 0:
                        self.highest_rank = current_index
                        self.best_method = method
                        break

    def __str__(self):
        return f"Results for {self.result_container.project_name}, Bug {self.result_container.bug_id}\n" + \
               f"Ranked {len(self.result_container.results)} Events and {len(self.result_methods)} Methods\n\n" + \
               f"There {'is one buggy function' if len(self.buggy_methods) == 1 else f'are {len(self.buggy_methods)} buggy functions'} in this commit: \n" + \
               f"{os.linesep.join(list(f'    {filename}: {method}, Lines {min(linenos)} -> {max(linenos)}' for filename, method, linenos in self.buggy_methods))}\n\n" + \
               f"Most suspicious method:\n    {str(self.result_methods[0])}\n\n" + \
               f"Most suspicious buggy method: Rank #{self.highest_rank + 1}, " + \
               f"Top {(self.highest_rank + 1) * 100.0 / len(self.result_methods)}%\n" + \
               f"    {str(self.best_method)}"


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
        print("\n\nTop 10 most suspicious events:\n")
        i = 0
        for r in evaluation.result_container.results[:10]:
            i += 1
            print(f"#{i}: {r}")
        sys.stdout = old_stdout
    os.system(f"less \"{result_dump}\"")  #
    print("Results have been written to " + result_dump)
