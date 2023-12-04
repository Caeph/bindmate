# TODO
from kmer.kmer_comparison_functions import *
from pathlib import Path

script_path = Path(__file__).absolute()
script_dir = os.path.split(script_path)[0]

# individual motifs modelled by hocomoco AND by ProBound
__motif_info = [
    # TODO
]

hoco_file = "H12CORE_pfms.txt"

# define normalization dataset:
bg_kmers = create_background(24, "random", bg_size=1500)

predefined_functions = {
    # name : metric object
    "lcs": LCSmetric(),
    "hoco_iou": HocomocoIOU(os.path.join(script_dir, "kmer", hoco_file),
                            1,
                            bg_kmers=bg_kmers),
    "probound_mse_human": ProBoundHumanMSE(bg_kmers=bg_kmers)
}