# TODO
from pathlib import Path
import os
from backgrounds.backgrounds import create_background
from kmer.kmer_comparison_functions import *

script_path = Path(__file__).absolute()
script_dir = os.path.split(script_path)[0]

hoco_file = "H12CORE_pfms.txt"
available_hoco_datasets = os.path.join(script_dir, "kmer", "sources_for_motif_based_scoring",
                                                   "hocomoco", "datasets.csv")


def initialize_available_functions(k, use_motifs_individually, background_type, background_size, bgkwargs):
    # define normalization dataset:
    if background_type is not None:
        bg_kmers = create_background(k, background_type, bg_size=background_size,
                                 **bgkwargs
                                 )
    else:
        bg_kmers = None

    # general functions
    predefined_functions = {
        "lcs": LCSmetric(),
        "gc": GCcontent(),
        "pair": PairContent(),
        "shape:EP" : ShapeMetric("EP"),
        "shape:HelT": ShapeMetric("HelT"),
        "shape:MGW": ShapeMetric("MGW"),
        "shape:ProT": ShapeMetric("ProT"),
        "shape:Roll": ShapeMetric("Roll"),
    }

    if not use_motifs_individually:
        predefined_functions["probound_mse_human"] = ProBoundHumanMSE(bg_kmers=bg_kmers)

        with open(available_hoco_datasets) as dataset_reader:
            next(dataset_reader)  # skip header
            for line in dataset_reader:
                datasetname, species, _, motifs = line.strip("\n").split(",")
                motifs = motifs.split(";")

                # IOU
                function = HocomocoIOU(
                    os.path.join(script_dir, "kmer", hoco_file),
                    1,
                    bg_kmers=bg_kmers,
                    selected_motifs=motifs
                )
                func_title = f"hoco_iou_{species.lower()}_{datasetname.lower().replace(' ', '_')}"
                predefined_functions[func_title] = function

                # MSE
                function = HocomocoMSE(
                    os.path.join(script_dir, "kmer", hoco_file),
                    bg_kmers=bg_kmers,
                    selected_motifs=motifs
                )
                func_title = f"hoco_mse_{species.lower()}_{datasetname.lower().replace(' ', '_')}"
                predefined_functions[func_title] = function
    else:
        raise NotImplementedError("TODO - individual motifs")
        # TODO individual models for a motif selection

    return predefined_functions
