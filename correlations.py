from Evaluator.Ranking import EVENT_TYPES
from Evaluator.Evaluation import Evaluation
from Evaluator.CombiningMethod import avg
import numpy as np


class EvaluationProfile:
    def __init__(self, ev: Evaluation):
        self.ranking_profiles = ev.ranking_infos.copy()
        self.num_rankings = len(self.ranking_profiles)
        self.avg_num_events = avg(list(p.len_events for p in self.ranking_profiles))
        self.avg_num_events_by_type = {t: avg(list(p.num_events_by_type[t] for p in self.ranking_profiles)) for t in EVENT_TYPES}
        self.avg_len_ranking = avg(list(p.len_methods for p in self.ranking_profiles))
        self.avg_evaluation_metrics = ev.evaluation_metrics
        self.avg_unique_lines_covered = avg(list(p.unique_lines_covered for p in self.ranking_profiles))
        self.avg_num_tests = avg(list(p.num_tests for p in self.ranking_profiles))
        self.avg_num_tests_passed = avg(list(p.num_tests_passed for p in self.ranking_profiles))
        self.avg_num_tests_failed = avg(list(p.num_tests_failed for p in self.ranking_profiles))
        self.avg_covered_lines_per_test = avg(list(p.covered_lines_per_test for p in self.ranking_profiles))

    def get_correlation_matrix(self):
        arr_num_events = np.array(list(p.len_events for p in self.ranking_profiles))
        arr_num_events_by_type = {t: np.array(list(p.num_events_by_type[t] for p in self.ranking_profiles)) for t in
                                       EVENT_TYPES}
        arr_len_ranking = np.array(list(p.len_methods for p in self.ranking_profiles))
        arr_evaluation_metrics = {i: {k: np.array(list(p.evaluation_metrics[k][i] for p in self.ranking_profiles)) for k in [1, 3, 5, 10]} for i in range(3)}
        arr_unique_lines_covered = np.array(list(p.unique_lines_covered for p in self.ranking_profiles))
        arr_num_tests = np.array(list(p.num_tests for p in self.ranking_profiles))
        arr_num_tests_passed = np.array(list(p.num_tests_passed for p in self.ranking_profiles))
        arr_num_tests_failed = np.array(list(p.num_tests_failed for p in self.ranking_profiles))
        arr_covered_lines_per_test = np.array(list(p.covered_lines_per_test for p in self.ranking_profiles))
        datasets = [
            arr_num_events,
            arr_len_ranking,
            arr_unique_lines_covered,
            arr_num_tests,
            arr_num_tests_passed,
            arr_num_tests_failed,
            arr_covered_lines_per_test,
        ]
        r = np.corrcoef(datasets)
        print(r)

