from functions import *
# THIS IS FOR DEVELOPMENT ONLY

# to comment out once development is done
import input_loading
from kmer_to_kmer_matchscores import PairingResults
from seq_to_seq_matchscores import *

# sqdf = input_loading.load_fasta_input("../../noblurr_human_genome_sample_bigger.fasta")
# sqdf = input_loading.load_fasta_input("../../noblurr_balanced_human_genome.fasta")
# sqdf = input_loading.load_fasta_input("../../small_unbalanced_test_dataset_randombg.fasta")
# sqdf = input_loading.load_fasta_input("../../fake_sequence_less_blurred_0_10_250_100_100.fasta")

# metrics = [
#     # "lcs",
#     "hoco_iou",
#     "probound_mse_human"
# ]
#
# results = calculate_kmer_to_kmer_matchscores(sqdf, 24, metrics)
# results.save("../../kmer_pairing_result_test")
#
# results = PairingResults.load("../../kmer_pairing_result_test")


def main():
    inputfile = "../../small_unbalanced_test_dataset_randombg.fasta"  # swift, for general functionality testing
    # inputfile = "../../fake_sequence_less_blurred_0_10_250_100_100.fasta"

    out = "../../test_results_match_probabilities"
    tool = PairingProbabilityCalculator(24, [
        "lcs",
     #   "hoco_iou",
     #   "probound_mse_human"
                                                ])
    similarities = tool.fit_predict_fasta(inputfile)
    # similarities.save(out)

    # similarities = PairingResults.load(out)

    seq2seq = calculate_seq_to_seq_similarities(similarities)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
