from functions import *
from kmer_to_kmer_matchscores import PairingResults


def main():
    # inputfile = "../../small_unbalanced_test_dataset_randombg.fasta"  # swift, for general functionality testing
    # inputfile = "../../fake_sequence_less_blurred_0_10_250_100_100.fasta"
    #
    out = "../../test_results_match_probabilities"
    # tool = PairingProbabilityCalculator(24, [
    #                             "lcs",
    #                             "hoco_iou",
    #                             "probound_mse_human"
    #                         ], out)
    # similarities = tool.fit_predict_fasta(inputfile)  # seq to seq
    # with open(os.path.join(out, "seq2seq_results.csv"), mode='w') as writer:
    #     for item in similarities:
    #         print(item.to_string(), file=writer)
    analysis = AnalyticalTool.load(out)
    print()






if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
