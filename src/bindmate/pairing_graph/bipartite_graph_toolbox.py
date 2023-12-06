import numpy as np
import networkx as nx


def create_bipartite_graph(viable):
    G = nx.Graph()
    if len(viable) == 0:
        return G

    ebunch = [(u, v, dict(edge_type='between', proba=proba)) for u, v, proba in zip(
        viable['kmer_i1'].values,
        viable['kmer_i2'].values,
        viable['probability'].values)]
    G.add_edges_from(ebunch)
    return G


def find_best_pairing(orig_G):
    G = orig_G.copy()

    if len(G.edges()) == 0:
        return SeqToSeqPairing(0, None, None)

    pairing = nx.max_weight_matching(G, weight='proba')
    pairing_cost = np.sum([orig_G.edges[u, v]['proba'] for u, v in pairing])
    return SeqToSeqPairing(len(pairing), pairing_cost, pairing)


class SeqToSeqPairing:
    def __init__(self, len_pairing, pairing_cost, pairing):
        self.pairing = pairing
        self.len = len_pairing
        self.pairing_cost = pairing_cost

    def to_string(self):
        paired_identifiers = ",".join([f"{x}--{y}" for x,y in self.pairing])
        desc = f"{self.pairing_cost};{self.len};{paired_identifiers}"
        return desc

    @staticmethod
    def load(desc_string):
        pairing_cost, length, paired = desc_string.split(";")
        pairing_cost = float(pairing_cost)
        length = int(length)
