from functions import *
import os
import sys

if __name__ == '__main__':
    script_dir = os.path.split(os.path.realpath(__file__))[0]
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    os.makedirs(output_path, exist_ok=True)

    tool = PairingProbabilityCalculator(42,
                                        [
                                            # 'hoco_mse_human_basic_helix-loop-helix_factors_(bhlh)',
                                            # 'hoco_mse_human_basic_leucine_zipper_factors_(bzip)',
                                            # 'hoco_mse_human_homeo_domain_factors',
                                            # 'hoco_mse_human_helix-turn-helix_domains',
                                            # 'hoco_mse_human_basic_domains',

                                            'hoco_mse_human_more_than_3_adjacent_zinc_fingers',
                                            'hoco_mse_human_hox-related',
                                            'hoco_mse_human_paired-related_hd',
                                            'hoco_mse_human_multiple_dispersed_zinc_fingers',
                                            'hoco_mse_human_nk-related',

                                            'hoco_mse_human_three-zinc_finger_kruppel-related',
                                            'hoco_mse_human_ets-related',
                                            'gc',
                                            'pair',
                                            "probound_mse_human",

                                            'shape:EP',
                                            'shape:HelT',
                                            'shape:MGW',
                                            'shape:ProT',
                                            'shape:Roll',
                                        ],
                                        output_path,
                                        background_type="sampled",
                                        background_size=1500,
                                        max_em_step=50,
                                        threads=1,
                                        no_gmm_models=3,
                                        preselection_part=0.2,
                                        no_matched_models=2,
                                        bootstrap_no=20,
                                        feature_size=10,
                                        background_source_file=os.path.join(script_dir, 'backgrounds/upstream2000.fa'),
                                        )
    similarities = tool.fit_predict_fasta(input_path)  # seq to seq
