from Evaluator.Ranking import EVENT_TYPES
from Evaluator.Evaluation import Evaluation
from Evaluator.CombiningMethod import avg
from evaluate_multi import EvaluationRun
from scipy.stats import rankdata
import numpy as np
import matplotlib.pyplot as plt

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
    for i, id in enumerate(datasets["App ID"]):
        ax.scatter(datasets[x_metric][i], datasets[y_metric][i], s=5, alpha=1, color=category_colors[id])
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
    print(r)
    if plot:
        fig, ax = plt.subplots()
        im = ax.imshow(r)
        im.set_clim(-1, 1)
        ax.grid(False)
        plt.xticks(np.arange(len(datasets.items())), datasets.keys(), rotation='vertical')
        ax.yaxis.set(ticks=np.arange(len(datasets.items())), ticklabels=(datasets.keys()))
        for i in range(len(datasets)):
            for j in range(len(datasets)):
                ax.text(j, i,  str(r[i, j])[:4], ha='center', va='center',
                        color='r')
        cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
        plt.show()
    return r

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

    def add_evaluation(self, ev: Evaluation):
        self.ranking_profiles.extend(ev.ranking_infos)

    def get_datasets(self):
        arr_num_events = np.array(list(p.len_events for p in self.ranking_profiles))
        arr_num_events_by_type = {t: np.array(list(p.num_events_by_type[t] for p in self.ranking_profiles)) for t in
                                  EVENT_TYPES}
        arr_len_ranking = np.array(list(p.len_methods for p in self.ranking_profiles))
        arr_evaluation_metrics = {
            i: {k: np.array(list(p.evaluation_metrics[k][i] for p in self.ranking_profiles)) for k in [1, 3, 5, 10]} for
            i in range(3)}
        arr_unique_lines_covered = np.array(list(p.unique_lines_covered for p in self.ranking_profiles))
        arr_num_tests = np.array(list(p.num_tests for p in self.ranking_profiles))
        arr_num_tests_passed = np.array(list(p.num_tests_passed for p in self.ranking_profiles))
        arr_num_tests_failed = np.array(list(p.num_tests_failed for p in self.ranking_profiles))
        arr_covered_lines_per_test = np.array(list(p.covered_lines_per_test for p in self.ranking_profiles))
        arr_unique_values_in_top_10 = np.array(list(10 - p.top_10_suspiciousness_value_ties for p in self.ranking_profiles))
        arr_loc = np.array(list(LOC_BY_APP[p.info.project_name] for p in self.ranking_profiles))
        arr_coverage_fraction = np.array(list(p.unique_lines_covered/LOC_BY_APP[p.info.project_name] for p in self.ranking_profiles))
        apps = list(LOC_BY_APP.keys())
        arr_app_id = np.array(list(apps.index(p.info.project_name) for p in self.ranking_profiles))
        datasets = {
            "Num events": arr_num_events,
            "Num methods": arr_len_ranking,
            "Num LOC Covered": arr_unique_lines_covered,
            "Num LOC total": arr_loc,
            "Coverage": arr_coverage_fraction,
            "Num test": arr_num_tests,
            "Num tests_p": arr_num_tests_passed,
            "Num tests_f": arr_num_tests_failed,
            "Num lc_per_test": arr_covered_lines_per_test,
            "Num unq. in t10": arr_unique_values_in_top_10,
            "App ID": arr_app_id,
        }
        for t in EVENT_TYPES:
            datasets.update({f"Num {t.__name__}": arr_num_events_by_type[t]})
        for i in range(3):
            for k in [1, 3, 5, 10]:
                datasets.update({f"res_t{i}_k{k}": arr_evaluation_metrics[i][k]})
        metric_avgs = []
        for p in self.ranking_profiles:
            a_1 = {i: avg(list(p.evaluation_metrics[k][i] for k in [1, 3, 5, 10])) for i in range(3)}
            metric_avgs.append(avg(list(a_1[i] for i in range(3))))
        datasets.update({'metric_avgs': np.array(metric_avgs)})
        return datasets



