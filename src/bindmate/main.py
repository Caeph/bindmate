from functions import *
from kmer_to_kmer_matchscores import PairingResults


def main():
    inputfile = "../../small_unbalanced_test_dataset_randombg.fasta"  # swift, for general functionality testing
    # inputfile = "../../fake_sequence_less_blurred_0_10_250_100_100.fasta"
    #
    # inputfile = "../../biodata_CTCF_TP53_l:300_n:200:200.fasta"
    # out = "../../test_results_match_probabilities"
    # out = "../../test_results_biodata_convergence_crit"
    out = "../../test_results_biodata_gmm"
    tool = PairingProbabilityCalculator(42,
                                        [
                                            # 'hoco_mse_human_basic_helix-loop-helix_factors_(bhlh)',
                                            # 'hoco_mse_human_basic_leucine_zipper_factors_(bzip)',
                                            # 'hoco_mse_human_homeo_domain_factors',
                                            # 'hoco_mse_human_helix-turn-helix_domains',
                                            # 'hoco_mse_human_basic_domains',
                                            'pair',
                                            'gc',
                                            # 'shape:EP',
                                            # 'shape:HelT',
                                            # 'shape:MGW',
                                            # 'shape:ProT',
                                            # 'shape:Roll',
                                            # "probound_mse_human"
                                        ],
                                        out, background_type="sampled", background_size=1500,
                                        background_source_file='backgrounds/upstream2000.fa',
                                        no_matched_models=2,
                                        max_em_step=10,
                                        bootstrap_no=3,
                                        preselection_part=0.25,
                                        threads=8
                                        )
    similarities = tool.fit_predict_fasta(inputfile)  # seq to seq

    # optimized = PairingResults.load(out)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
