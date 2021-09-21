import itertools

from Evaluator.Ranking import EVENT_TYPES
from Evaluator.Evaluation import Evaluation
from Evaluator.RankerEvent import *
from Evaluator.CombiningMethod import *
from Evaluator.SimilarityCoefficient import *
from evaluate_multi import EvaluationRun
from scipy.stats import rankdata
import numpy as np
import matplotlib.pyplot as plt

selected_events = [LineCoveredEvent, SDBranchEvent, SDReturnValueEvent, AbsoluteScalarValueEvent, AbsoluteReturnValueEvent]

LOC_BY_APP = {
    'ansible': 207300,
    'cookiecutter': 4700,
    'PySnooper': 4300,
    'spacy': 102000,
    'sanic': 14100,
    'httpie': 5600,
    'keras': 48200,
    'matplotlib': 213200,
    'thefuck': 1741,
    'pandas': 292200,
    'black': 96000,
    'scrapy': 30700,
    'luigi': 41500,
    'fastapi': 25300,
    'tornado': 27700,
    'tqdm': 4800,
    'youtube-dl': 124500,
}


def scatter_plot(datasets, x_metric: str, y_metric: str, x_title: str, y_title: str):
    category_colors = plt.get_cmap('brg')(
        np.linspace(0.05, 0.95, len(LOC_BY_APP.items())))
    fig, ax = plt.subplots()
    for i, name in enumerate(LOC_BY_APP.keys()):
        print(f"{name} - {category_colors[i]}")
    for i in range(len(datasets[x_metric])):
        ax.scatter(datasets[x_metric][i], datasets[y_metric][i], s=30, alpha=.3, color=category_colors[datasets["App ID"][i]])
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    fig.set_size_inches(4, 3)
    plt.show()


def get_correlation_matrix(datasets, plot=False, rank_based=False):
    if rank_based:
        datasets = datasets.copy()
        for k in datasets.keys():
            datasets[k] = rankdata(datasets[k])
    r = np.corrcoef(list(datasets.values()))
    to_print = np.tri(r.shape[0], k=-1).astype(np.bool)
    r = np.ma.array(r, mask=to_print)
    print(r)
    if plot:
        fig, ax = plt.subplots()
        im = ax.imshow(r)
        im.set_clim(-1, 1)
        ax.grid(False)
        plt.xticks(np.arange(len(datasets.items())), datasets.keys(), fontsize=8, rotation='vertical')
        plt.yticks(np.arange(len(datasets.items())), datasets.keys(), fontsize=8)
        for i in range(len(datasets)):
            for j in range(len(datasets)):
                ax.text(j, i, '{:.2f}'.format(float(r[i, j])) if str(r[i, j]) != "--" else "", ha='center', va='center',
                        color='r', fontsize=12)
        cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
        plt.show()
    return r

class EvaluationProfile:
    def __init__(self, ev: Evaluation):
        self.ranking_profiles = list(sorted(ev.ranking_infos.copy(), key=lambda ri: (ri.project_name, ri.bug_id)))
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

    def add_evaluation(self, ev: Evaluation):
        self.ranking_profiles.extend(ev.ranking_infos)

    def get_datasets(self):
        arr_num_events = np.array(list(p.len_events for p in self.ranking_profiles))
        arr_num_sus_events = np.array(list(p.len_events_sus for p in self.ranking_profiles))
        arr_num_events_by_type = {t: np.array(list(p.num_events_by_type[t] for p in self.ranking_profiles)) for t in
                                  EVENT_TYPES}
        arr_len_ranking = np.array(list(p.len_methods for p in self.ranking_profiles))
        arr_evaluation_metrics = {
            i: {k: np.array(list(p.evaluation_metrics[k][i] for p in self.ranking_profiles)) for k in [1, 3, 5, 10]} for
            i in range(3)}
        arr_unique_lines_covered = np.array(list(p.unique_lines_covered for p in self.ranking_profiles))
        arr_num_tests = np.array(list(p.num_tests for p in self.ranking_profiles))
        arr_buggy_methods = np.array(list(p.num_buggy_methods for p in self.ranking_profiles))
        arr_num_tests_passed = np.array(list(p.num_tests_passed for p in self.ranking_profiles))
        arr_num_tests_failed = np.array(list(p.num_tests_failed for p in self.ranking_profiles))
        arr_covered_lines_per_test = np.array(list(p.covered_lines_per_test for p in self.ranking_profiles))
        arr_total_lines_per_test = np.array(list(LOC_BY_APP[p.info.project_name] / p.num_tests for p in self.ranking_profiles))
        arr_unique_values_in_top_10 = np.array(list(10 - p.top_10_suspiciousness_value_ties for p in self.ranking_profiles))
        arr_loc = np.array(list(LOC_BY_APP[p.info.project_name] for p in self.ranking_profiles))
        arr_coverage_fraction = np.array(list(p.unique_lines_covered/LOC_BY_APP[p.info.project_name] for p in self.ranking_profiles))
        arr_coverage_fraction = (arr_coverage_fraction - min(arr_coverage_fraction)) / (max(arr_coverage_fraction) - min(arr_coverage_fraction))
        apps = list(LOC_BY_APP.keys())
        arr_app_id = np.array(list(apps.index(p.info.project_name) for p in self.ranking_profiles))
        arr_sum_num_events = np.array(list(p.sum_num_events for p in self.ranking_profiles))
        arr_sum_events_passed = np.array(list(p.sum_events_passed for p in self.ranking_profiles))
        arr_sum_events_failed = np.array(list(p.sum_events_failed for p in self.ranking_profiles))
        arr_unq_events_passed = np.array(list(p.sum_unique_events_passed for p in self.ranking_profiles))
        arr_unq_events_failed = np.array(list(p.sum_unique_events_failed for p in self.ranking_profiles))
        arr_methods_sus = np.array(list(p.len_methods_susp for p in self.ranking_profiles))
        arr_methods_unsus = np.array(list(p.len_methods_unsusp for p in self.ranking_profiles))
        arr_evt_once = np.array(list(p.num_events_only_covered_by_one_test for p in self.ranking_profiles))
        arr_frac_evt_once = np.array(list(p.num_events_only_covered_by_one_test / p.len_events for p in self.ranking_profiles))
        arr_evt_only_f = np.array(list(p.num_events_only_covered_by_failed_tests for p in self.ranking_profiles))
        arr_crs_cvg = np.array(list(p.lines_covered_more_than_once for p in self.ranking_profiles))
        arr_frac_covered_m = np.array(list((p.lines_covered_more_than_once / p.num_events_by_type[LineCoveredEvent]) for p in self.ranking_profiles))
        datasets = dict()
        datasets = {
            #"Num events": arr_num_events,
            #"Frac events once": arr_frac_evt_once,
            "Sum num events": arr_sum_num_events,
            "Sus events": arr_num_sus_events,
            "Sum num events passed": arr_sum_events_passed,
            "Sum num events failed": arr_sum_events_failed,
            #"L cov. m. t. once": arr_crs_cvg,
            #"Frac. L cov. m. t. once": arr_frac_covered_m,
            #"Unq events passed": arr_unq_events_passed,
            "Unq events failed": arr_unq_events_failed,
            #"Events ol. recd. once": arr_evt_once,
            #"Events ol. recd. once fail.": arr_evt_only_f,
            #"Num methods": arr_len_ranking,
            #"Num methods sus": arr_methods_sus,
            #"Num methods unsus": arr_methods_unsus,
            #"Num LOC Covered": arr_unique_lines_covered,
            #"Num LOC total": arr_loc,
            #"Coverage": arr_coverage_fraction,
            "Num test": arr_num_tests,
            #"Num tests_p": arr_num_tests_passed,
            #"Num tests_f": arr_num_tests_failed,
            #"Num lc_per_test": arr_covered_lines_per_test,
            #"Num loc_per_test": arr_total_lines_per_test,
            #"Num unq. in t10": arr_unique_values_in_top_10,
            "App ID": arr_app_id,
            "Ratio buggy methods": arr_buggy_methods / arr_len_ranking,
            #"Avg. method len": np.nan_to_num(arr_num_events_by_type[LineCoveredEvent]/arr_len_ranking, nan=0.0, posinf=0.0),
            #f"Ratio {LineCoveredEvent.__name__}": np.nan_to_num(arr_num_events_by_type[LineCoveredEvent] / arr_num_events, nan=0.0, posinf=999999999)
        }
        for t in EVENT_TYPES:
            datasets.update({f"Ratio {t.__name__}": np.nan_to_num(arr_num_events_by_type[t] / arr_sum_num_events, nan=0.0, posinf=999999999)})
        for t in EVENT_TYPES:
            datasets.update({f"LOC ratio {t.__name__}": arr_num_events_by_type[t] / arr_loc})
        #for t in EVENT_TYPES:
        #    datasets.update({f"frac sus {t.__name__}": np.array(list(p.num_sus_events_by_type[t]/(p.len_events_sus if p.len_events_sus > 0 else 999999999999) for p in self.ranking_profiles))})
        #for i in range(3):
        #    for k in [1, 3, 5, 10]:
        #        datasets.update({f"res_t{i}_k{k}": arr_evaluation_metrics[i][k]})
        metric_avgs = []
        for p in self.ranking_profiles:
            a_1 = {i: avg(list(p.evaluation_metrics[k][i] for k in [1, 3, 5, 10])) for i in range(3)}
            metric_avgs.append(avg(list(a_1[i] for i in range(3))))
        datasets.update({'metric_avgs': np.array(metric_avgs)})
        return datasets


def get_best_ris_by_type(run: EvaluationRun):
    evs = list(filter(
        lambda e: len(e.combining_method.event_types) == 1 and e.combining_method.event_types[0] in selected_events,
        run.evaluations))
    best_ris_by_type = {t: [] for t in selected_events}
    ri_to_avg = lambda ri: avg(list(avg(list(ri.evaluation_metrics[k][i] for k in [1, 3, 5, 10])) for i in range(3)))
    for i in range(len(evs[0].ranking_infos)):
        avgs = {e: ri_to_avg(e.ranking_infos[i]) for e in evs}
        best_e, _ = sorted(avgs.items(), key=lambda v: v[1], reverse=True)[0]
        best_ris_by_type[best_e.combining_method.event_types[0]].append(best_e.ranking_infos[i])


def extend_w_lc_best(datasets, run: EvaluationRun):
    evs = list(filter(
        lambda e: len(e.combining_method.event_types) == 1 and e.combining_method.event_types[0] in selected_events,
        run.evaluations))
    ri_to_avg = lambda ri: avg(list(avg(list(ri.evaluation_metrics[k][i] for k in [1, 3, 5, 10])) for i in range(3)))
    datasets['lc_best'] = []
    for i in range(len(evs[0].ranking_infos)):
        avgs = {e: ri_to_avg(sorted(e.ranking_infos, key=lambda ri: (ri.project_name, ri.bug_id))[i]) for e in evs}
        best_e, _ = sorted(avgs.items(), key=lambda v: v[1], reverse=True)[0]
        datasets['lc_best'].append(best_e.combining_method.event_types[0] == LineCoveredEvent)


def extend_w_relative_performance(datasets, run: EvaluationRun):
    evs = list(filter(
        lambda e: len(e.combining_method.event_types) == 1 and e.combining_method.event_types[0] in selected_events,
        run.evaluations))
    for t in selected_events:
        datasets[f"Relative {t.__name__}"] = []
    ri_to_avg = lambda ri: avg(list(avg(list(ri.evaluation_metrics[k][i] for k in [1, 3, 5, 10])) for i in range(3)))
    for i in range(len(evs[0].ranking_infos)):
        avgs = {e.combining_method.event_types[0]: ri_to_avg(sorted(e.ranking_infos, key=lambda ri: (ri.project_name, ri.bug_id))[i]) for e in evs}
        for t in selected_events:
            avg_all = np.average(list(avgs.values()))
            datasets[f"Relative {t.__name__}"].append(avgs[t]/avg_all if avg_all > 0 else 1)


def extend_w_event_type_specific_results(datasets, evs: EvaluationRun):
    methods = {ev.combining_method.event_types[0] for ev in evs.evaluations}
    for ev in evs.evaluations:
        if not isinstance(ev.combining_method, FilteredCombiningMethod):
            continue
        if len(ev.combining_method.event_types) > 1:
            continue
        metric_avgs = []
        for p in list(sorted(ev.ranking_infos.copy(), key=lambda ri: (ri.project_name, ri.bug_id))):
            a_1 = {i: avg(list(p.evaluation_metrics[k][i] for k in [1, 3, 5, 10])) for i in range(3)}
            metric_avgs.append(avg(list(a_1[i] for i in range(3))))
        datasets[ev.combining_method.event_types[0].__name__.replace("Return", "Return-\n").replace("Scalar", "Scalar-\n")] = np.array(metric_avgs)
    #for m1, m2 in itertools.combinations(methods, 2):
    #    datasets[f"Diff. metrics  {m1.__name__} | {m2.__name__}"] = datasets[f"Avg. metrics only {m1.__name__}"] - datasets[f"Avg. metrics only {m2.__name__}"]


if __name__ == '__main__':
    evr = EvaluationRun.load("results_evaluation/event_type_combinations2_single.pickle.gz")
    datasets = EvaluationProfile(evr.evaluations[0]).get_datasets()
    extend_w_event_type_specific_results(datasets, evr)
    extend_w_relative_performance(datasets, evr)
    get_correlation_matrix(datasets, plot=True, rank_based=False)