import ast
from _ast import While, If
from typing import Tuple, List, Dict, Any
from ast import NodeVisitor
from .Methods import DebuggerMethod
from .utils import getValidProjectDir


class BranchInstruction:
    def __init__(self, method_object: DebuggerMethod, location: Tuple[str, str, int], first_body_lineno: int, while_loop: bool = False, breaks: List[Tuple[str, str, int]] = None):
        self.method_object = method_object
        self.location = location
        self.first_body_lineno = first_body_lineno
        self.while_loop = while_loop
        self.breaks = breaks

    def __str__(self):
        return f"{'while' if self.while_loop else 'if'} @ {str(self.location)}"


class BranchExtractor(NodeVisitor):
    def __init__(self, filename: str, method_objects: Dict[Tuple[str, str, int], DebuggerMethod], *args, **kwargs):
        super(BranchExtractor, self).__init__(*args, **kwargs)
        self.filename = filename
        _method_objects = filter(lambda e: e[1].file == filename, method_objects.items())
        self.method_objects_by_lineno = {lineno: method for (fn, mn, lineno), method in _method_objects}
        self.branches = list()

    def visit_If(self, node: If) -> Any:
        if node.lineno in self.method_objects_by_lineno.keys():
            location = (self.filename, self.method_objects_by_lineno[node.lineno].name, node.lineno)
            self.branches.append(BranchInstruction(self.method_objects_by_lineno[node.lineno], location, node.body[0].lineno))

    def visit_While(self, node: While) -> Any:
        if node.lineno in self.method_objects_by_lineno.keys():
            location = (self.filename, self.method_objects_by_lineno[node.lineno].name, node.lineno)
            breaks = list()
            for n in ast.walk(node):
                if isinstance(n, ast.Break) and n.lineno in self.method_objects_by_lineno.keys():
                    breaks.append((self.filename, self.method_objects_by_lineno[n.lineno].name, n.lineno))
            self.branches.append(BranchInstruction(self.method_objects_by_lineno[node.lineno], location, node.body[0].lineno, True, breaks))


def extractBranchesFromCode(_results, info, method_objects):
    directory = getValidProjectDir(_results, info, instrument=True)
    files = {mo.file for mo in method_objects.values()}
    branches = list()
    for filename in files:
        with open(directory + "/" + filename, "rt") as f:
            parsed = ast.parse(f.read())
        extractor = BranchExtractor(filename, method_objects)
        extractor.visit(parsed)
        branches.extend(extractor.branches)
    return branches
