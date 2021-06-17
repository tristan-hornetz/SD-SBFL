import ast
import inspect
import os
import sys

import pkg_resources

from typing import List, Tuple, Set, Dict

installed = {pkg.key for pkg in pkg_resources.working_set}
if 'unidiff' in installed:
    from unidiff import PatchSet, PatchedFile
else:
    PatchSet = PatchedFile = object


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
        self.work_dir = _results.work_dir + "/" + _results.project_name

    def __str__(self):
        ret = ""
        for a in self.attrs:
            ret += f"{a}=\"{getattr(self, a)}\"\n"
        return ret



class LineNumberExtractor(ast.NodeVisitor):
    """
    NodeVisitor that stores the line numbers of nodes it visits
    """

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
        return hash(self.file + "\\" + self.name + "\\" + ';'.join(str(n) for n in sorted(self.linenos)))

    def __str__(self):
        return f"{self.file}[{self.name} | Lines {';'.join(str(n) for n in sorted(self.linenos))}]" + \
               (f" - Suspiciousness: {self.suspiciousness}" if hasattr(self, 'suspiciousness') else "")

    def __eq__(self, other):
        if not isinstance(other, DebuggerMethod):
            return False
        return str(self) == str(other)


def getMethodStringFromEvent(event: str, work_dir):
    return next(reversed(event.split(" @ "))).replace(work_dir + "/", "")


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

    parent_elements = filter(lambda node: isinstance(node, ast.FunctionDef)
                                          or isinstance(node, ast.AsyncFunctionDef),
                             parent_dict[line_nodes[0]])

    parents = list(parent_elements)
    if len(parents) == 0:
        return None
    extractor = LineNumberExtractor()
    extractor.visit(next(reversed(parents)))
    return next(reversed(parents)).name, tuple(sorted(extractor.linenos))


def extractMethodsFromFile(directory: str, file: str, methods: Dict[str, Set[int]]) -> List[Tuple[str, Set[int]]]:
    """
    Extract the lines of methods from a specified file
    :param directory: The directory in which the file is located
    :param file: The name of the file
    :param methods: The names of the methods to extract, mapped to the lines encountered during testing
    :return: A list of tuples, representing the extracted methods
    """
    if os.path.exists(directory + "/" + file):
        with open(directory + "/" + file, "rt") as f:
            node = ast.parse(f.read())
        function_defs = set(
            filter(lambda n: isinstance(n, ast.FunctionDef) or isinstance(n, ast.AsyncFunctionDef), ast.walk(node)))
        lines_per_method = list()
        for d in filter(lambda n: n.name in methods.keys(), function_defs):

            extractor = LineNumberExtractor()
            extractor.visit(d)
            if len(methods[d.name].intersection(extractor.linenos)) > 0:
                lines_per_method.append((d.name, extractor.linenos))
        return lines_per_method
    return list()


def getBuggyMethodsFromFile(project_root: str, file: PatchedFile, is_target_file: bool) -> List[Tuple[str, Set[int]]]:
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

    names = set(filter(lambda o: o is not None, (getParentFunctionFromLineNo(source, lineno) for lineno in linenos)))
    extracted_methods = dict()
    for name, linenos in names:
        if name in extracted_methods.keys():
            extracted_methods[name].update(linenos)
        else:
            extracted_methods[name] = set(linenos)
    return extractMethodsFromFile(project_root, file.path, extracted_methods)


def getValidProjectDir(_results, fixed=False, directory="", instrument=False):
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
    root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    binary_dir = root_dir + '/_BugsInPy/framework/bin'
    os.system(
        f'{binary_dir}/bugsinpy-checkout -p {_results.project_name} -i {_results.bug_id} -v {1 if fixed else 0} -w {directory}')
    if instrument:
        debugger_module = os.path.abspath(root_dir + '/run_test.py')
        os.system(
            f'{binary_dir}/bugsinpy-instrument -f -c {debugger_module} -w {directory + "/" + _results.project_name}')

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
    repo_dir = getValidProjectDir(_results, False, instrument=True)
    for file in patch_set.modified_files:
        methods.update(
            (file.path, method, tuple(linenos)) for method, linenos in getBuggyMethodsFromFile(repo_dir, file, False))
    repo_dir = getValidProjectDir(_results, True, instrument=True)
    for file in patch_set.modified_files:
        methods.update(
            (file.path, method, tuple(linenos)) for method, linenos in getBuggyMethodsFromFile(repo_dir, file, True))
    return list(DebuggerMethod(name, path, set(linenos)) for path, name, linenos in methods)


def extractMethodsFromCode(_results, info: BugInfo) -> List[DebuggerMethod]:
    """
    Extract information about methods encountered during a test run
    The extracted info can be used for verification purposes
    :param _results: The SFL_Results of a test-run
    :param info: A BugInfo instance for a test-run
    :return: A list of DebuggerMethod instances representing the methods as extracted from code
    """
    methods_per_file = dict()  # Dict[str, Dict[str, Set[int]]]
    lines_encountered_per_method_str = dict()
    for event, lineno in _results.results:
        method_str = getMethodStringFromEvent(event, info.work_dir)
        if method_str in lines_encountered_per_method_str.keys():
            lines_encountered_per_method_str[method_str].add(lineno)
        else:
            lines_encountered_per_method_str[method_str] = {lineno}

    for s in lines_encountered_per_method_str.keys():
        m_arr = s.split("[")
        if len(m_arr) > 1:
            if m_arr[0] in methods_per_file.keys():
                if m_arr[1].split("]")[0] in methods_per_file[m_arr[0]].keys():
                    methods_per_file[m_arr[0]][m_arr[1].split("]")[0]].update(lines_encountered_per_method_str[s])
                else:
                    methods_per_file[m_arr[0]][m_arr[1].split("]")[0]] = lines_encountered_per_method_str[s]
            else:
                methods_per_file[m_arr[0]] = {m_arr[1].split("]")[0]: lines_encountered_per_method_str[s]}
    directory = getValidProjectDir(_results, False, instrument=True)
    method_objects = set()
    for file in methods_per_file.keys():
        lines_per_method = extractMethodsFromFile(directory, file, methods_per_file[file])
        for method, linenos in lines_per_method:
            method_objects.add(DebuggerMethod(method, file, linenos))
    return list(method_objects)