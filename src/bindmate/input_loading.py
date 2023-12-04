import pandas as pd
from pyfastaq.sequences import file_reader as fasta_reader
import swifter


def load_fasta_input(filename, subset=None):
    df = pd.DataFrame([[entry.id, entry.seq] for entry in fasta_reader(filename)], columns=['header', 'sequence'])

    if subset is not None:
        df = df.sample(subset)
    return df


def load_bed_input(filename, fasta_input_filename, subset=None):
    ...
    # TODO