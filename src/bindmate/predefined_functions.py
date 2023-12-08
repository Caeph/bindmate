# TODO
from pathlib import Path
import os
from backgrounds.backgrounds import create_background

from kmer.kmer_comparison_functions import *

script_path = Path(__file__).absolute()
script_dir = os.path.split(script_path)[0]

hoco_file = "H12CORE_pfms.txt"


def initialize_available_functions(k, background_type, background_size, bgkwargs):
    # define normalization dataset:
    if background_type is not None:
        bg_kmers = create_background(k, background_type, bg_size=background_size,
                                 **bgkwargs
                                 )
    else:
        bg_kmers = None

    predefined_functions = {
        "lcs": LCSmetric(),
        "hoco_iou": HocomocoIOU(os.path.join(script_dir, "kmer", hoco_file),
                                1,
                                bg_kmers=bg_kmers),
        "probound_mse_human": ProBoundHumanMSE(bg_kmers=bg_kmers)
    }
    return predefined_functions
