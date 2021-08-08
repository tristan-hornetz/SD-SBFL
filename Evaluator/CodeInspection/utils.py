import ast
import inspect
import os
import sys
from shutil import copy

from git import Repo
from unidiff import PatchSet


def get_info_directory(_results):
    """
    Get the absolute path of the BugsInPy directory containing information on the bug tested in _results
    :param _results: The SFL_Results of a test run
    :return: The absolute path of the BugsInPy directory containing information on the bug tested in _results
    """
    base = os.path.dirname(os.path.realpath(sys.argv[0]))
    return os.path.abspath(f"{base}/_BugsInPy/projects/{_results.project_name}/bugs/{_results.bug_id}")


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
        self.project_name = _results.project_name
        self.bug_id = _results.bug_id

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


def getPatchSet(info: BugInfo):
    """
    Create a unidiff PatchSet instance from the bug_patch.txt file of a specific Bug
    :param info: A BugInfo instance for the bug to be loaded
    :return: A unidiff PatchSet instance from the bug_patch.txt file of the bug from info
    """
    with open(info.info_dir + "/bug_patch.txt", "rt") as f:
        diff_text = f.read()
    return PatchSet(diff_text)


def mkdirRecursive(directory: str):
    if os.path.exists(directory):
        return
    parent = directory.rsplit("/", 1)[0]
    mkdirRecursive(parent)
    os.mkdir(directory)


def getCleanRepo(_results, info: BugInfo, directory):
    if not os.path.exists(directory):
        mkdirRecursive(directory)
    try:
        repo = Repo(directory)
        repo.git.add('--all')
        repo.git.stash()
        repo.git.checkout(info.buggy_commit_id)
    except:
        os.system(f"rm -rf \"{directory}\"")
        repo = Repo.clone_from(info.project_github_url, directory)
        repo.git.checkout(info.buggy_commit_id)
    return repo

def extra_instrumentation(_results, info: BugInfo, directory):
    """
    Perform project-specific instrumentation steps to replicate the effects of alltest.sh
    :param _results: The SFL_Results of the test run
    :param info: The BugInfo for _resultd
    :param directory: The directory to create the repo in
    """
    if _results.project_name == "sanic":
        os.system(f"ln -s $(readlink -f {directory})/sanic $(readlink -f {directory})/tests/sanic")


def getValidProjectDir(_results, info: BugInfo, fixed=False, directory="", instrument=False):
    """
    Create a valid Git Repo for a specific bug and return the path to it
    :param _results: The SFL_Results of the test run
    :param info: The BugInfo for _resultd
    :param fixed: Checkout the fixed commit?
    :param directory: The directory to create the repo in
    :return: The path to a valid git repo for the bug in info
    """
    if directory == "":
        directory = os.path.dirname(inspect.getfile(getCleanRepo)) + "/../.temp/" + _results.project_name

    repo = getCleanRepo(_results, info, directory)
    if fixed:
        repo.git.checkout(info.fixed_commit_id)

    root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    binary_dir = root_dir + '/_BugsInPy/framework/bin'
    if instrument:
        for file in os.scandir(info.info_dir):
            if os.path.isfile(str(file.path)):
                copy(str(file.path), directory + "/bugsinpy_" + str(file.path).replace(info.info_dir + "/", ""))
        debugger_module = os.path.abspath(root_dir + '/run_single_test.py')
        os.system(
            f'{binary_dir}/bugsinpy-instrument -f -c {debugger_module} -w {directory} > /dev/null 2>&1')
        extra_instrumentation(_results, info, directory)

    return directory
