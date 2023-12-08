import numpy as np
from pyfastaq.sequences import file_reader as fasta_reader
import pandas as pd

bases = list("ACGT")


##### background creation
def create_background(k, background_type, bg_size, **kwargs):
    if background_type == "random":
        bgkmers = np.array(["".join(np.random.choice(bases, size=k)) for _ in range(bg_size)])
    elif background_type == "sampled":
        to_sample_from_file = kwargs['background_source_file']  # fasta expected
        print(f"Sampling background from {to_sample_from_file}.")
        bg_df = pd.DataFrame([[entry.id, entry.seq] for entry in fasta_reader(to_sample_from_file)],
                             columns=["id", "seq"])
        bg_df = bg_df[bg_df['seq'].str.len() > k]
        if len(bg_df) >= bg_size:
            bg_df = bg_df.sample(bg_size)

        def sample_random_kmer(sequence):
            l = len(sequence) - k
            start = np.random.randint(l)
            return sequence[start:start+k]

        bg_df['kmer'] = bg_df['seq'].swifter.progress_bar(False).apply(sample_random_kmer).str.upper()
        bgkmers = bg_df['kmer'].values
    else:
        raise NotImplementedError("No other background type is implemented.")

    return bgkmers
