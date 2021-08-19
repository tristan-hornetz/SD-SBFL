import ast
import os
from typing import Dict, Set, List, Tuple

from unidiff import PatchedFile

from .utils import BugInfo, getPatchSet, getValidProjectDir, LineNumberExtractor


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
        self.suspiciousness = []

    def add_lineno(self, lineno):
        self.linenos.add(lineno)

    def __hash__(self):
        return hash(self.file + "\\" + self.name + "\\" + ';'.join(str(n) for n in sorted(self.linenos)))

    def __str__(self):
        if len(self.suspiciousness) == 0:
            self.suspiciousness = [-1]
        return f"{self.file}[{self.name} | Lines {';'.join(str(n) for n in sorted(self.linenos))}]"

    def __eq__(self, other):
        if not isinstance(other, DebuggerMethod):
            return False
        return str(self) == str(other)


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
    :param file: The name of the file
    :param methods: The names of the methods to extract, mapped to the lines encountered during testing
    :return: A list of tuples, representing the extracted methods
    """
    if os.path.exists(directory + "/" + file):
        with open(directory + "/" + file, "rt") as f:
            try:
                node = ast.parse(f.read())
            except SyntaxError:
                print(f"Syntax Error in {directory + '/' + file}")
                return list()
        function_defs = set(
            filter(lambda n: isinstance(n, ast.FunctionDef) or isinstance(n, ast.AsyncFunctionDef), ast.walk(node)))
        lines_per_method = list()
        for d in filter(lambda n: n.name in methods.keys(), function_defs):
            extractor = LineNumberExtractor()
            extractor.visit(d)
            if len(methods[d.name].intersection(extractor.linenos)) > 0:
                lines_per_method.append((d.name, extractor.linenos))

        #compensate for methods which could not be extracted
        found = {n for n, _ in lines_per_method}
        for n, ls in methods:
            if n not in found:
                lines_per_method.append((n, ls))

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

    #TODO

    return extractMethodsFromFile(project_root, file.path, extracted_methods)


def getBuggyMethods(_results, info: BugInfo):
    """
    Get a list of all methods that are were modified to fix a specific bug
    :param _results: The SFL_Results of the test run
    :param info: The BugInfo for _results
    :return: A list of all methods that are were modified to fix a specific bug
    """
    patch_set = getPatchSet(info)
    fixed_state_methods = set()

    repo_dir = getValidProjectDir(_results, info, True, instrument=True)

    for file in patch_set.modified_files:
        fixed_state_methods.update(
            (file.path, method, tuple(linenos)) for method, linenos in getBuggyMethodsFromFile(repo_dir, file, True))

    repo_dir = getValidProjectDir(_results, info, False, instrument=True)
    fixed_state_files = {m[0] for m in fixed_state_methods}
    buggy_state_methods = set()
    for f in fixed_state_files:
        methods = filter(lambda m: m[0] == f, fixed_state_methods)
        buggy_state_methods.update((f, method, tuple(linenos)) for method, linenos in
                                   extractMethodsFromFile(repo_dir, f, {m[1]: set(m[2]) for m in methods}))

    for file in patch_set.modified_files:
        buggy_state_methods.update(
            (file.path, method, tuple(linenos)) for method, linenos in getBuggyMethodsFromFile(repo_dir, file, False))
    return list(DebuggerMethod(name, path, set(linenos)) for path, name, linenos in buggy_state_methods)


def extractMethodsFromCode(_results, info: BugInfo) -> Dict[Tuple[str, str, int], DebuggerMethod]:
    """
    Extract information about methods encountered during a test run
    The extracted info can be used for verification purposes
    :param _results: The SFL_Results of a test-run
    :param info: A BugInfo instance for a test-run
    :return: A dict of type (<filename>, <method-name>, <lineno>) -> DebuggerMethod
    """
    methods_per_file = dict()  # Dict[str, Dict[str, Set[int]]]
    directory = getValidProjectDir(_results, info, False, instrument=True)
    for event in _results.results:
        _filename, method_name, lineno, *other = event
        filename = _filename.replace(info.work_dir + "/", "")
        if filename in methods_per_file.keys():
            if method_name in methods_per_file[filename].keys():
                methods_per_file[filename][method_name].add(lineno)
            else:
                methods_per_file[filename][method_name] = {lineno}
        else:
            methods_per_file[filename] = {method_name: {lineno}}
    method_objects = dict()
    for file in methods_per_file.keys():
        lines_per_method = extractMethodsFromFile(directory, file, methods_per_file[file])
        for method, linenos in lines_per_method:
            method_object = DebuggerMethod(method, file, linenos)
            for lineno in linenos:
                method_objects[info.work_dir + "/" + file, method, lineno] = method_object
    return method_objects
