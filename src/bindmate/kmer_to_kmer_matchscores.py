import time

import numpy as np
import pandas as pd

from kmer.make_kmers import *
from predefined_functions import predefined_functions
from itertools import combinations
from joblib import Parallel, delayed, parallel_config, cpu_count
from em_algorithm.optimization import *
import swifter
import multiprocessing


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
    # metric_values = parallel_apply_along_axis(calculate, 1, kmer_combinations)
    # chunks = np.array_split(kmer_combinations, cpus)
    #
    # def calculate(r):
    #     i1, i2 = r
    #     return metric.compare_kmers(unique_kmers[i1], unique_kmers[i2])
    #
    # def do_chunk(chunk):
    #     return np.apply_along_axis(calculate, 1, chunk)
    #
    # with parallel_config(backend='threading', n_jobs=cpus):
    #     metric_chunks = Parallel()(
    #              delayed(do_chunk)(chunk) for chunk in chunks)
    # return np.hstack(metric_chunks)
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
    # return np.array(np.meshgrid(np.meshgrid(indices, indices))).T.reshape(-1, 2)
    return np.array(np.meshgrid(indices, indices))


def __optimize(all_ranks, full_metrics, priors=None, max_step=10, tolerance=0.0001):
    matched = ProbabilityModel(1, full_metrics)
    unmatched = ProbabilityModel(0, full_metrics)

    if priors is None:
        priors = {0: 0.95, 1: 0.05}

    em_algo = EMOptimizer(
        possible_latent=[0, 1],
        priors=priors,
        models={0: unmatched, 1: matched}
    )
    models = em_algo.optimize(all_ranks, max_step, tolerance)
    probas_0 = unmatched.calculate_probability(all_ranks)
    probas_1 = matched.calculate_probability(all_ranks)
    return probas_0, probas_1


def __store(save_results_path):
    # TODO
    raise NotImplementedError("Saving to file")


def __preselect(probas_0, probas_1, part):
    if part >= 1:
        return np.arange(0, len(probas_0))

    no = int(part * len(probas_0))
    one_thr = probas_1[np.argsort(-probas_1)[no]]
    zero_thr = probas_0[np.argsort(probas_0)[no]]
    return np.where((probas_1 >= one_thr) & (probas_0 <= zero_thr))[0]


class PairingResults:
    def __init__(self, unique_kmers, kmer_combinations, probas_1, probas_0, mapped_kmers):
        self.unique_kmers = unique_kmers
        self.available_combinations = pd.DataFrame(kmer_combinations, columns=["kmer_i1", "kmer_i2"]
                                                   ).reset_index().rename(columns={"index" : "combination_index"})  # indexes
        self.probas_1 = probas_1
        self.probas_0 = probas_0
        self.mapped_kmers_df = mapped_kmers[["input_sequence_index", "kmer_index"]]

    def calculate_for_seq_index_pair(self, seqi1, seqi2):
        seqi1_df = self.mapped_kmers_df[self.mapped_kmers_df['input_sequence_index'] == seqi1]
        seqi2_df = self.mapped_kmers_df[self.mapped_kmers_df['input_sequence_index'] == seqi2]

        cross_df = pd.merge(seqi1_df, seqi2_df, how='cross', suffixes=("_1", "_2"))
        pairs = pd.merge(cross_df, self.available_combinations,
                     left_on=['kmer_index_1', "kmer_index_2"], right_on=["kmer_i1", "kmer_i2"],
                         how="inner").drop(
                     columns=["kmer_i1", "kmer_i2"])

        pairs['probability'] = pairs['combination_index'].apply(lambda i: self.probas_1[i])
        return pairs


def __calculate_kmer_to_kmer_matchscores(unique_kmers, kmers_mapped_to_sqs,
                                         full_metrics, cpus, save_results, preselection_part):
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

    pairwise_ranks = np.vstack(rank_results).T  # combine rank scores to single object
    # shape: [no of combinations, no of metrics]

    # todo optional zapsani do souboru, jake jsou metric values, jake jsou rank values
    if save_results is not None:
        __store(save_results)

    del rank_results, metric_values, rank_values  # cleanup

    # em algo for score calculation
    probas_0, probas_1 = __optimize(pairwise_ranks, full_metrics)
    print(f"Probabilities calculated, optimization complete: {time.time() - start}")
    start = time.time()

    # ranks are no longer needed
    del pairwise_ranks

    # preselection, nechat mapability!!!
    selected_indices = __preselect(probas_0, probas_1, preselection_part)
    print(f"Preselection done: {time.time() - start}")
    start = time.time()

    selected_kmer_combinations = kmer_combinations[selected_indices, :]
    selected_proba_0 = probas_0[selected_indices]
    selected_proba_1 = probas_1[selected_indices]

    results = PairingResults(
        unique_kmers, selected_kmer_combinations, selected_proba_0, selected_proba_1,
        kmers_mapped_to_sqs
    )
    return results


def calculate_kmer_to_kmer_matchscores(inputdf, k, metrics,
                                       cpus=-1, save_results=None, preselection_part=0.5):
    # curate the metrics
    if cpus == -1:
        cpus = cpu_count()

    # read input kmers
    print(f"Calculating on {cpus} CPUs.")
    unique_kmers, kmers_mapped_to_sqs = __read_unique_kmers(inputdf, k)

    full_metrics = []
    for m in metrics:
        if m in predefined_functions:
            full = predefined_functions[m]
            full.initialize(unique_kmers)
            print(f"Metric {full.name} initialized.")
        else:
            # TODO -- user defined or stupid
            raise NotImplementedError("This function is not implementented.")

        full_metrics.append(full)
    # full metrics is a list of metric dictionaries

    # do the work
    pairwise_scoring_results = __calculate_kmer_to_kmer_matchscores(
        unique_kmers, kmers_mapped_to_sqs, full_metrics, cpus, save_results, preselection_part)
    return pairwise_scoring_results


# to comment out once development is done
import input_loading

# sqdf = input_loading.load_fasta_input("../../noblurr_human_genome_sample_bigger.fasta")
# sqdf = input_loading.load_fasta_input("../../noblurr_balanced_human_genome.fasta")
sqdf = input_loading.load_fasta_input("../../small_unbalanced_test_dataset_randombg.fasta")
# sqdf = input_loading.load_fasta_input("../../fake_sequence_less_blurred_0_10_250_100_100.fasta")

metrics = [
    "lcs",
    "hoco_iou",
    "probound_mse_human"
]

results = calculate_kmer_to_kmer_matchscores(sqdf, 24, metrics)
results.calculate_for_seq_index_pair(1,0)
