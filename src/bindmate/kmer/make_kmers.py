import pandas as pd
import numpy as np
from pyfastaq.sequences import file_reader as fasta_reader
import swifter


def make_kmers(seq, k, offset):
    """
    Get k-mers from a seq with an offset. No sliding window, k-mers are exclusive.
    Only words of size k are returned (excess trimmed)
    :param seq:
    :param k:
    :param offset:
    :return:
    """
    excess = (len(seq) - offset) % k
    if excess != 0:
        s = seq[offset:-excess]
    else:
        s = seq[offset:]
    kmers = np.apply_along_axis(lambda x: "".join(x), 1, np.reshape(np.array(list(s)), (-1, k)))
    return kmers


def get_overlapping_kmers(seq, k, kmer_overlapping=2):
    part = k // kmer_overlapping
    offsets = [i*part for i in range(kmer_overlapping)]
    for offset in offsets:
        kmers = make_kmers(seq, k, offset)
        yield kmers


