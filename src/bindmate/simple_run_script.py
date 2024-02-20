from functions import *
import os
import sys

if __name__ == '__main__':
    script_dir = os.path.split(os.path.realpath(__file__))[0]
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    os.makedirs(output_path, exist_ok=True)

    tool = PairingProbabilityCalculator(24,
                                        [  # "lcs",
                                            'hoco_mse_human_basic_helix-loop-helix_factors_(bhlh)',
                                            'hoco_mse_human_basic_leucine_zipper_factors_(bzip)',
                                            'hoco_mse_human_homeo_domain_factors',
                                            'hoco_mse_human_helix-turn-helix_domains',
                                            'hoco_mse_human_basic_domains',
                                            'gc',
                                            'shape:EP',
                                            'shape:HelT',
                                            'shape:MGW',
                                            'shape:ProT',
                                            'shape:Roll',
                                            "probound_mse_human"
                                        ],
                                        output_path,
                                        background_type="sampled",
                                        background_size=1500,
                                        max_em_step=20,
                                        preselection_part=0.5,
                                        no_matched_models=2,
                                        background_source_file=os.path.join(script_dir,
                                                                            'backgrounds/upstream2000.fa'),
                                        )
    similarities = tool.fit_predict_fasta(input_path)  # seq to seq
