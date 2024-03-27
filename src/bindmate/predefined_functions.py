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


def initialize_available_functions(k, use_motifs_individually, no_models, background_type, background_size, bgkwargs):
    # define normalization dataset:
    if background_type is not None:
        bg_kmers = create_background(k, background_type, bg_size=background_size,
                                 **bgkwargs
                                 )
    else:
        bg_kmers = None

    # general functions
    predefined_functions = {
        "lcs": LCSmetric(no_models),
        "gc": GCcontent(no_models),
        "pair": PairContent(no_models),
        "shape:EP" : ShapeMetric("EP", no_models),
        "shape:HelT": ShapeMetric("HelT", no_models),
        "shape:MGW": ShapeMetric("MGW", no_models),
        "shape:ProT": ShapeMetric("ProT", no_models),
        "shape:Roll": ShapeMetric("Roll", no_models),
    }

    if not use_motifs_individually:
        # TODO make possible different no_models for each metric
        predefined_functions["probound_mse_human"] = ProBoundHumanMSE(no_models, bg_kmers=bg_kmers)

        with open(available_hoco_datasets) as dataset_reader:
            next(dataset_reader)  # skip header
            for line in dataset_reader:
                datasetname, species, _, motifs = line.strip("\n").split(",")
                motifs = motifs.split(";")

                # IOU
                function = HocomocoIOU(
                    os.path.join(script_dir, "kmer", hoco_file),
                    1,
                    no_models,
                    bg_kmers=bg_kmers,
                    selected_motifs=motifs
                )
                func_title = f"hoco_iou_{species.lower()}_{datasetname.lower().replace(' ', '_')}"
                predefined_functions[func_title] = function

                # MSE
                function = HocomocoMSE(
                    os.path.join(script_dir, "kmer", hoco_file),
                    no_models,
                    bg_kmers=bg_kmers,
                    selected_motifs=motifs
                )
                func_title = f"hoco_mse_{species.lower()}_{datasetname.lower().replace(' ', '_')}"
                predefined_functions[func_title] = function
    else:
        raise NotImplementedError("TODO - individual motifs")
        # TODO individual models for a motif selection

    return predefined_functions
