import _ast
import ast
from typing import Any, List


class MethodInfoExtractor(ast.NodeVisitor):
    def __init__(self):
        self.type_counts = dict()
        self.variable_names = dict()
        self.subnode_count = 0
        self.num_expressions = 0

    def visit(self, node: _ast.AST) -> Any:
        self.subnode_count += 1
        t = type(node)
        if t not in self.type_counts.keys():
            self.type_counts[t] = 0
        self.type_counts[t] += 1
        if isinstance(node, _ast.expr):
            self.num_expressions += 1
            if isinstance(node, _ast.Name):
                if node.id not in self.variable_names.keys():
                    self.variable_names[node.id] = 0
                self.variable_names[node.id] += 1
        return super().visit(node)


class StatisticsExtractor:
    def __init__(self, methods, all_types: List[type]):
        infos: List[MethodInfoExtractor] = list(m.method_info for m in methods)
        self.types = all_types
        self.method_count = len(infos)
        self.absolute_type_counts = {
            t: sum(info.type_counts[t] if t in info.type_counts.keys() else 0 for info in infos) for t in all_types}
        self.average_type_counts = {t: self.absolute_type_counts[t] / self.method_count for t in all_types}
        self.average_subnode_count = sum(info.subnode_count for info in infos) / self.method_count
        self.average_expression_count = sum(info.num_expressions for info in infos) / self.method_count
        self.average_statement_count = sum(
            info.subnode_count - info.num_expressions for info in infos) / self.method_count
        self.unique_names_per_method = sum(len(info.variable_names.keys()) for info in infos) / self.method_count
        self.name_references_per_method = sum(sum(info.variable_names.values()) for info in infos) / self.method_count


class CodeStatistics:
    def __init__(self, all_methods, buggy_methods):
        all_types = list(set.union(*(m.method_info.type_counts.keys() for m in all_methods)))
        self.buggy = StatisticsExtractor(buggy_methods, all_types)
        self.all = StatisticsExtractor(all_methods, all_types)
