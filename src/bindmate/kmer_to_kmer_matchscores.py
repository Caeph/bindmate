import os
import time

import numpy as np
import pandas as pd

from kmer.make_kmers import *
from predefined_functions import initialize_available_functions
from itertools import combinations
from joblib import Parallel, delayed, parallel_config, cpu_count
from em_algorithm.optimization import *
import swifter
import multiprocessing

import dask.array as da


def __read_unique_kmers(input_sqs, k):
    input_sqs['kmers'] = input_sqs['sequence'].swifter.progress_bar(False).apply(
        lambda s: list(get_overlapping_kmers(s, k)))

    input_sqs['window'] = [[0, 1] for _ in range(len(input_sqs))]
    two_lined = input_sqs.reset_index().explode(['kmers', 'window'])
    two_lined['order'] = two_lined['kmers'].apply(lambda lst: np.arange(len(lst)))

    two_lined = two_lined.rename(columns={"index": "input_sequence_index"})

    kmers_mapped_to_sqs = two_lined.explode(['kmers', 'order']).drop(columns=['sequence']).reset_index().drop(
        columns=['index'])
    unique_kmers = kmers_mapped_to_sqs["kmers"].unique()
    mapper = {kmer: i for i, kmer in enumerate(unique_kmers)}

    kmers_mapped_to_sqs['kmer_index'] = kmers_mapped_to_sqs['kmers'].map(mapper)

    return unique_kmers, kmers_mapped_to_sqs


def __calculate_all_to_all_metric(kmer_combinations, unique_kmers, metric, cpus):
    # with parallel_config(backend='threading', n_jobs=cpus):
    #     metric_values = Parallel()(
    #         delayed(metric.compare_kmers)(unique_kmers[i1], unique_kmers[i2]) for i1, i2 in kmer_combinations)
    def calculate(r):
        i1, i2 = r
        return metric.compare_kmers(i1, i2)

    metric_values = np.apply_along_axis(calculate, 0, kmer_combinations)
    # metric_values = da.apply_along_axis(calculate, 0, kmer_combinations,
    #                                      shape=(kmer_combinations.shape[1:]), dtype=float).compute()
    return metric_values


def __translate_metric_to_rank(value_array, metric_type, cpus):
    unique_values = np.unique(value_array)
    if metric_type == 'similarity':
        order = np.argsort(-unique_values)
    elif metric_type == 'distance':
        order = np.argsort(unique_values)
    else:
        raise NotImplementedError("Unknown metric type.")

    mapper = {unique_values[item]: i for i, item in enumerate(order)}

    def vec_translate(a, my_dict):
        return np.vectorize(my_dict.__getitem__)(a)

    rank_values = vec_translate(value_array, mapper)
    return np.array(rank_values)


def __get_kmer_combinations(unique_kmers):
    n = len(unique_kmers)
    indices = np.arange(n)
    # dask_array = da.from_array(np.array(np.meshgrid(indices, indices)), chunks=100)  # leaving chunks='auto'
    # return dask_array.persist()
    return np.array(np.meshgrid(indices, indices))


def __optimize(all_ranks, full_metrics, priors=None, max_step=10, tolerance=0.0001, em_params_file=None):
    print()
    matched = ProbabilityModel(1, full_metrics)
    unmatched = ProbabilityModel(0, full_metrics)

    if priors is None:
        priors = {0: 0.95, 1: 0.05}

    em_algo = EMOptimizer(
        possible_latent=[0, 1],
        priors=priors,
        models={0: unmatched, 1: matched}
    )
    print("Initialization complete...")
    if em_params_file is None:
        models = em_algo.optimize(all_ranks, max_step, tolerance)
    else:
        with open(em_params_file, mode='w') as writer:
            print('unmatched_params,matched_params,prior_0,prior_1', file=writer)
            models = em_algo.optimize(all_ranks, max_step, tolerance, parameter_colector=writer)
    print("Calculating final probability")
    probas_0 = models[0].calculate_probability(all_ranks)
    probas_1 = models[1].calculate_probability(all_ranks)

    # calculate match probability given metrics
    prior_0, prior_1 = [em_algo.priors[z] for z in [0, 1]]
    metrics_proba = probas_0 * prior_0 + probas_1 * prior_1
    match_given_metrics = probas_1 * prior_1 / metrics_proba
    mismatch_given_metrics = probas_0 * prior_0 / metrics_proba

    return mismatch_given_metrics, match_given_metrics


def __store(save_results_path, pairwise_ranks, kmer_combinations, metrics):
    results = pd.DataFrame(np.hstack([kmer_combinations, pairwise_ranks]),
                           columns=[
                               "kmer_index_1", "kmer_index_2",*[m.name for m in metrics]
                                    ]
                           )
    results.to_csv(save_results_path, index=False)


def __preselect(full_probas_0, full_probas_1, part, kmer_combinations):
    if part >= 1:
        return np.arange(0, len(full_probas_0))

    without_symmetry = kmer_combinations[:, 0] > kmer_combinations[:, 1]

    probas_0 = full_probas_0[without_symmetry]
    probas_1 = full_probas_1[without_symmetry]
    no = int(part * len(probas_0))
    one_thr = probas_1[np.argsort(-probas_1)[no]]
    zero_thr = probas_0[np.argsort(probas_0)[no]]

    return np.where((full_probas_1 >= one_thr) & (full_probas_0 <= zero_thr))[0]


class PairingResults:
    def __init__(self, unique_kmers, kmer_combinations, probas_1, probas_0, mapped_kmers):
        self.unique_kmers = unique_kmers  # numpy array
        self.available_combinations = pd.DataFrame(kmer_combinations, columns=["kmer_i1", "kmer_i2"]
                                                   ).reset_index().rename(
            columns={"index": "combination_index"})  # indexes, pandas df
        self.probas_1 = probas_1  # np array
        self.probas_0 = probas_0  # np array
        self.mapped_kmers_df = mapped_kmers[["input_sequence_index", "kmer_index"]]  # pandas df

    def get_all_seq_indices(self):
        return self.mapped_kmers_df['input_sequence_index'].unique()

    # def calculate_for_seq_index_pair(self, seqi1, seqi2):
    #     # THIS IS SLOW!!!!!
    #     seqi1_df = self.mapped_kmers_df[self.mapped_kmers_df['input_sequence_index'] == seqi1]
    #     seqi2_df = self.mapped_kmers_df[self.mapped_kmers_df['input_sequence_index'] == seqi2]
    #
    #     cross_df = pd.merge(seqi1_df, seqi2_df, how='cross', suffixes=("_1", "_2"))
    #     pairs = pd.merge(cross_df, self.available_combinations,
    #                      left_on=['kmer_index_1', "kmer_index_2"], right_on=["kmer_i1", "kmer_i2"],
    #                      how="inner").drop(
    #         columns=["kmer_i1", "kmer_i2"])
    #
    #     pairs['probability'] = pairs['combination_index'].apply(lambda i: self.probas_1[i])
    #     return pairs

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.available_combinations.to_csv(os.path.join(path, "available_combinations.csv.gz"),
                                           index=False)
        self.mapped_kmers_df.to_csv(os.path.join(path, "mapped_kmers_df.csv.gz"),
                                    index=False)
        for array, core_filename in zip(
                [self.probas_1, self.probas_0],
                ["probas_1", "probas_0"]
        ):
            np.save(os.path.join(path, core_filename + ".npy"), array)

        np.savetxt(os.path.join(path, "unique_kmers.txt"), self.unique_kmers, fmt="%s")

    @staticmethod
    def load(path):
        unique_kmers = np.loadtxt(os.path.join(path, "unique_kmers.txt"), dtype=str)
        kmer_combinations = pd.read_csv(os.path.join(path, "available_combinations.csv.gz"))
        probas_1 = np.load(os.path.join(path, "probas_1.npy"))
        probas_0 = np.load(os.path.join(path, "probas_0.npy"))
        mapped_kmers = pd.read_csv(os.path.join(path, "mapped_kmers_df.csv.gz"))
        return PairingResults(unique_kmers, kmer_combinations, probas_1, probas_0, mapped_kmers)


def __calculate_kmer_to_kmer_matchscores(unique_kmers, kmers_mapped_to_sqs,
                                         full_metrics, cpus, save_results, preselection_part,
                                         max_em_step, em_params_file):
    start = time.time()

    # calculation of metric ranks
    kmer_combinations = __get_kmer_combinations(unique_kmers)  # indices only
    print(f"Unique kmer combinations created: {time.time() - start}")
    start = time.time()
    rank_results = []

    for metric in full_metrics:
        metric_values = __calculate_all_to_all_metric(kmer_combinations, unique_kmers, metric, cpus)
        print(f"Metric {metric.name} values calculated: {time.time() - start}")
        start = time.time()

        rank_values = __translate_metric_to_rank(metric_values, metric.get_type(), cpus)
        # reshape
        rank_values = np.reshape(rank_values, (-1, 1)).flatten().T

        rank_results.append(rank_values)
        print(f"Metric {metric.name} ranks calculated: {time.time() - start}")
        start = time.time()

    # reshape kmer_combinations
    kmer_combinations = np.vstack([
        np.reshape(kmer_combinations[0, :, :], (-1, 1)).flatten(),
        np.reshape(kmer_combinations[1, :, :], (-1, 1)).flatten()
    ]).T
    print("Combination indices reshaped.")

    pairwise_ranks = np.vstack(rank_results).T  # combine rank scores to single object
    # shape: [no of combinations, no of metrics]
    print("Ranks combined.")

    if save_results is not None:
        __store(save_results, pairwise_ranks, kmer_combinations, full_metrics)
        print("Results saved.")

    del rank_results, metric_values, rank_values  # cleanup
    print("Cleanup done.")

    # em algo for score calculation
    print("Starting optimization...")
    mismatch_proba, match_proba = __optimize(pairwise_ranks, full_metrics,
                                    max_step=max_em_step, em_params_file=em_params_file)
    print(f"Probabilities calculated, optimization complete: {time.time() - start}")
    start = time.time()

    # ranks are no longer needed
    del pairwise_ranks

    # preselection, keep mapability!!!
    selected_indices = __preselect(mismatch_proba, match_proba, preselection_part, kmer_combinations)
    print(f"Preselection done: {time.time() - start}")
    start = time.time()

    selected_kmer_combinations = kmer_combinations[selected_indices, :]
    selected_mismatch_proba = mismatch_proba[selected_indices]
    selected_match_proba = match_proba[selected_indices]

    #  unique_kmers, kmer_combinations, probas_1, probas_0, mapped_kmers
    results = PairingResults(
        unique_kmers, selected_kmer_combinations, selected_match_proba, selected_mismatch_proba,
        kmers_mapped_to_sqs
    )
    return results


def calculate_kmer_to_kmer_matchscores(inputdf, k, metrics, background_info,
                                       cpus=-1, max_em_step=20,
                                       save_results='../../test_results_match_probabilities/test_store.csv',
                                       preselection_part=0.5, em_params_file=None):
    # curate the metrics
    if cpus == -1:
        cpus = cpu_count()

    predefined_functions = initialize_available_functions(
        k, *background_info
    )

    # read input kmers
    unique_kmers, kmers_mapped_to_sqs = __read_unique_kmers(inputdf, k)

    full_metrics = []
    for m in metrics:
        if m in predefined_functions:
            start = time.time()
            full = predefined_functions[m]
            full.initialize(unique_kmers)
            print(f"Metric {full.name} initialized: {np.round(time.time() - start, decimals=5)}")
        else:
            # TODO -- user defined or stupid
            raise NotImplementedError("This function is not implementented.")

        full_metrics.append(full)
    # full metrics is a list of metric dictionaries

    # do the work
    pairwise_scoring_results = __calculate_kmer_to_kmer_matchscores(
        unique_kmers, kmers_mapped_to_sqs, full_metrics, cpus, save_results, preselection_part,
        max_em_step, em_params_file)
    return pairwise_scoring_results
