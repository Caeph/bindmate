import numpy as np
import pandas as pd

from pairing_graph.bipartite_graph_toolbox import *
from multiprocessing import Pool
from tqdm import tqdm


# REQUIRE SYMMETRICITY


def calculate_seq_to_seq_similarities(kmer_match_proba_obj, threads=5):
    seq_ids = kmer_match_proba_obj.get_all_seq_indices()
    all_to_all = np.array(np.meshgrid(seq_ids, seq_ids))

    todo = np.triu(all_to_all)  # applied to final two
    todo = np.vstack([
        np.reshape(todo[0, :, :], (-1, 1)).flatten(),
        np.reshape(todo[1, :, :], (-1, 1)).flatten()
    ]).T
    todo = np.unique(todo, axis=0)
    todo = todo[todo[:, 0] != todo[:, 1]]  # don't compare the same sequences
    todo = pd.DataFrame(todo, columns=['seq1_index', 'seq2_index'])

    # TO SPEED UP - ONLY KEEPING I > J PAIRS OF SEQUENCES
    kmer_combinations = kmer_match_proba_obj.available_combinations
    seq_to_kmer = kmer_match_proba_obj.mapped_kmers_df

    # sequence pairs with mapped kmers
    print("Mapping sequences and kmer probabilities")
    X = pd.merge(todo, seq_to_kmer, left_on='seq1_index', right_on='input_sequence_index'
                 ).drop(columns=['input_sequence_index']).rename(columns={"kmer_index": "kmer_i1"})
    X = pd.merge(X, seq_to_kmer, left_on='seq2_index', right_on='input_sequence_index'
                 ).drop(columns=['input_sequence_index']).rename(columns={"kmer_index": "kmer_i2"})

    # filter to contain only available kmer combinations
    X = pd.merge(X, kmer_combinations)
    X['probability'] = X["combination_index"].apply(lambda ci: kmer_match_proba_obj.probas_1[ci])

    print("Creating graphs")
    graphs = []
    for item in tqdm(X.groupby(by=['seq1_index', 'seq2_index'])):
        name, group = item
        G = create_bipartite_graph(group)
        G.graph['name'] = name  # s1, s2 indices
        graphs.append(G)

    print("Calculating distance as perfect matching")

    if threads == 1:
        results = []
        for item in tqdm(graphs):
            r = find_best_pairing(item)
            results.append(r)
    else:
        no = len(graphs)
        with Pool(threads) as pool:
            results = tqdm(pool.imap(find_best_pairing, graphs),
                           total=no)
            results = list(results)

    return results
