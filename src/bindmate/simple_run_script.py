from functions import *
import os
import sys

if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    os.makedirs(output_path, exist_ok=True)

    tool = PairingProbabilityCalculator(24, [
        "lcs",
        "hoco_iou_human_full",
        "probound_mse_human"
    ], output_path, background_type="sampled", background_size=1500, max_em_step=50, preselection_part=0.25,
                                        background_source_file='backgrounds/upstream2000.fa',
                                        )
    similarities = tool.fit_predict_fasta(input_path)  # seq to seq
