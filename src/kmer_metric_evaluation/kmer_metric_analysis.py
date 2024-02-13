import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pyfastaq.sequences import file_reader as fasta_reader
import swifter
import os

import sys
sys.path.insert(1, '..')
sys.path.insert(2, '../bindmate')

import bindmate


def calculate_metrics(metric, kmer_set, a='CTCF', b='TP53', only_metric=False):
    unique_kmers = kmer_set['kmers']
    metric.initialize(unique_kmers)

    kmer_combinations = bindmate.kmer_to_kmer_matchscores.__get_kmer_combinations(
        unique_kmers)
    metric_values = bindmate.kmer_to_kmer_matchscores.__calculate_all_to_all_metric(
        kmer_combinations, unique_kmers, metric, -1)

    if only_metric:
        return metric_values

    metric_values = bindmate.kmer_to_kmer_matchscores.__translate_metric_to_rank(metric_values, metric.get_type(), -1)
    print(metric_values.shape)
    metric_values = metric_values + np.triu(metric_values + np.nan)

    # sns.heatmap(metric_values)
    # plt.show()

    metric_df = pd.DataFrame(metric_values).reset_index()
    metric_df = pd.melt(metric_df, id_vars='index')
    metric_df.columns = ['x', 'y', 'value']
    metric_df = pd.merge(metric_df, kmer_set[["type"]], left_on='x', right_index=True).rename(
        columns={"type": "type_x"})
    metric_df = pd.merge(metric_df, kmer_set[["type"]], left_on='y', right_index=True).rename(
        columns={"type": "type_y"})
    metric_df = metric_df[metric_df['x'] != metric_df['y']]
    metric_df['tmp'] = (metric_df['type_x'] == metric_df['type_y'])  # .astype(str)
    metric_df = metric_df.dropna()

    metric_df['type'] = 'different'

    metric_df.loc[metric_df['tmp'] & (metric_df["type_x"] == a), 'type'] = a
    metric_df.loc[metric_df['tmp'] & (metric_df["type_x"] == b), 'type'] = b
    metric_df = metric_df[['x', 'y', 'value', 'type']]
    return metric_df


k = 24
input_file = "../../biodata_CTCF_TP53_l:300_n:10:10.fasta"
test_set = pd.DataFrame(
    [[entry.id, entry.seq] for entry in fasta_reader(input_file)],
    columns=['id', 'seq']
)
test_set["type"] = test_set["id"].str.split('-').str[0]
test_set['kmers'] = test_set['seq'].apply(lambda x: list(bindmate.make_kmers.get_overlapping_kmers(x, k)))

kmer_set = test_set.drop(columns=['seq', 'id']).explode("kmers").explode("kmers").drop_duplicates()
kmer_set = kmer_set.groupby(by='kmers').agg({
    "type": lambda x : ",".join(list(set(x)))
}).reset_index()

m = bindmate.predefined_functions.ShapeMetric("EP")
metric_df = calculate_metrics(m, kmer_set)

sns.histplot(data=metric_df, x='value', hue='type', common_norm=False, stat='percent', multiple='dodge')
plt.show()
