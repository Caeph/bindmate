from functions import *
from kmer_to_kmer_matchscores import PairingResults


def main():
    # inputfile = "../../small_unbalanced_test_dataset_randombg.fasta"  # swift, for general functionality testing
    inputfile = "../../fake_sequence_less_blurred_0_10_250_100_100.fasta"
    #
    out = "../../test_results_match_probabilities"
    tool = PairingProbabilityCalculator(24, [
                                "lcs",
                                "hoco_iou_human_full",
                                # "hoco_iou_human_basic_domains",
                                "probound_mse_human"
                            ], out, background_type="sampled", background_size=1500,
                                        background_source_file='backgrounds/upstream2000.fa',
                                        )
    similarities = tool.fit_predict_fasta(inputfile)  # seq to seq

    # optimized = PairingResults.load(out)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
