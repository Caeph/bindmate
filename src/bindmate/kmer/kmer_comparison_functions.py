from collections import Counter
from difflib import SequenceMatcher
from itertools import product, chain

import numpy as np
from scipy.optimize import minimize_scalar
from scipy import stats
from Bio.Seq import Seq
import Bio.motifs as bmotifs
import pyProBound
from tqdm import tqdm
import os
import rpy2.robjects as robjects
import warnings

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
robjects.r('library(BiocManager)')

script_dir = os.path.split(os.path.realpath(__file__))[0]


def normalize(affinities, bg_affinities):
    loc, scale = bg_affinities.mean(), bg_affinities.std()
    return (affinities - loc) / scale


def score(sq, pssm):
    if pssm.length > len(sq):
        return np.nan
    return np.fmax(pssm.calculate(Seq(sq)).max(),
                   pssm.reverse_complement().calculate(Seq(sq)).max())


##### functions for EM optimization
def gaussian_argmax(observed_vector, model_qs, sigma_pseudocount=0.01):
    pies = np.sum(model_qs)
    best_mu = 1 / pies * np.sum(observed_vector * model_qs)
    best_sigma = np.sqrt(1 / pies * np.sum(((observed_vector - best_mu) ** 2) * model_qs))
    return best_mu, best_sigma + sigma_pseudocount


def geometric_argmax(observed_vector, model_qs):
    # theoretical does not exist, using numerical
    # Pr[X = k] = (1-p)^(k-1) * p
    # log(Pr[X = k]) = (k-1) log(1-p) + log(p)

    def negative_geom_proba_func(p):
        result = model_qs * (
                observed_vector * np.log(1 - p) + np.log(p)
        )
        result = - np.sum(result)
        return result

    m = minimize_scalar(negative_geom_proba_func, bounds=[0, 1])

    return [m.x]


def uniform_argmax(observed_vector, model_qs):
    return [np.max(observed_vector)]


def apply_gaussian_argmax(q, x):
    return gaussian_argmax(x, q)


def apply_geometric_argmax(q, x):
    return geometric_argmax(x, q)


def apply_uniform_argmax(q, x):
    return uniform_argmax(x, q)


def apply_gaussian_proba(val, params):
    return stats.norm.pdf(val, params[0], params[1])


def apply_geometric_proba(val, params):
    return stats.geom.pmf(val, params[0])


def apply_uniform_proba(val, params):
    return 1 / params[0]



def apply_gmm_proba(val, params):
    total_proba = 0
    params_counter = 0
    no_models = len(params) // 3
    for i in range(no_models):
        loc, scale, weight = params[params_counter], params[params_counter + 1], params[params_counter + 2]
        total_proba += weight*stats.norm.pdf(val, loc, scale)
        params_counter += 3
    return total_proba



distributions = {
    "univariate_gaussian": dict(
        argmax=apply_gaussian_argmax,
        proba=apply_gaussian_proba
    ),
    "univariate_geometric": dict(
        argmax=apply_geometric_argmax,
        proba=apply_geometric_proba,
    ),
    "univariate_uniform": dict(
        argmax=apply_uniform_argmax,
        proba=apply_uniform_proba,
    ),
    "univariate_gmm": dict(
        proba=apply_gmm_proba,
        bounds=None
    #     bounds=list(chain.from_iterable(([-1e3, 1e8], [1e-3, 1e5], [1e-3, 1-1e-3]) for _ in range(no_models)))
    #                                            # loc, scale, weight
    )
}


class Metric:
    def __init__(self, name, metric_type, description
                 ):
        self.name = name
        self.type = metric_type
        self.desc = description
        self.match_optim = None
        self.mismatch_optim = None
        self.unique_kmers = None

    def define_optimalization_params(self, match_dict, mismatch_dict):
        self.match_optim = match_dict
        self.mismatch_optim = mismatch_dict

    def get_name(self):
        return self.name

    def get_type(self):
        return self.type

    def get_description(self):
        return self.desc

    def get_optimalization_info(self, z):  # match is True
        if z == 1:
            return self.match_optim
        elif z == 0:
            return self.mismatch_optim
        raise NotImplementedError("Other model is not implemented.")

    def compare_kmers(self, i1, i2):
        return None

    def initialize(self, unique_kmers):
        self.unique_kmers = unique_kmers
        return


class LCSmetric(Metric):
    def __init__(self, no_models):
        super().__init__("LCS",
                         "similarity",
                         "Size of the longest common substring shared between two kmers")
        local_max = 10e4
        super().define_optimalization_params(
            # 1
            dict(proba=distributions['univariate_gmm']['proba'],
                 params_bounds=distributions['univariate_gmm']['bounds'],
                 initial_parameters=list(
                     chain.from_iterable((i * local_max / no_models, (10 * i + 1), 1/no_models) for i in range(
                         no_models)))
                 ),
            # 0
            dict(proba=distributions['univariate_gmm']['proba'],
                 params_bounds=distributions['univariate_gmm']['bounds'],
                 initial_parameters=list(
                     chain.from_iterable((i * local_max / no_models, 10 * no_models, 1/no_models) for i in range(
                         no_models)))
                 ),
        )

    def compare_kmers(self, i1, i2):
        kmer1, kmer2 = self.unique_kmers[i1], self.unique_kmers[i2]
        match = SequenceMatcher(None, kmer1, kmer2).find_longest_match()
        return match.size


class PairContent(Metric):
    def __init__(self, no_models):
        super().__init__("pair",
                         "distance",
                         "Difference in dinucleotide pair content")
        # TODO set params - should be poisson/exponentials or sth like that
        local_max = 100
        super().define_optimalization_params(
            # 1
            dict(proba=distributions['univariate_gmm']['proba'],
                 params_bounds=distributions['univariate_gmm']['bounds'],
                 initial_parameters=list(
                     chain.from_iterable((i * local_max / no_models, (10 * i + 1), 1 / no_models) for i in range(
                         no_models)))
                 ),
            # 0
            dict(proba=distributions['univariate_gmm']['proba'],
                 params_bounds=distributions['univariate_gmm']['bounds'],
                 initial_parameters=list(
                     chain.from_iterable((i * local_max / no_models, 10 * no_models, 1 / no_models) for i in range(
                         no_models)))
                 ),
        )
        self.pairs = None
        self.info = ["".join(pair) for pair in product(list('ACGT'), list('ACGT'))]

    def __characterize(self, seq):
        counts = Counter([seq[i:i + 2] for i in range(len(seq) - 1)])
        res = np.array([counts[pair] for pair in self.info], dtype=np.uint8)
        return res

    def initialize(self, unique_kmers):
        self.pairs = np.vstack([self.__characterize(x) for x in unique_kmers])

    def compare_kmers(self, i1, i2):
        a = self.pairs[i1]
        b = self.pairs[i2]
        diff = a - b
        return np.mean(diff * diff)


class GCcontent(Metric):
    def __init__(self, no_models):
        super().__init__("gc",
                         "distance",
                         "Difference in GC content")

        # TODO set params - should be poisson/exponentials or sth like that
        local_max=50
        super().define_optimalization_params(
            # 1
            dict(proba=distributions['univariate_gmm']['proba'],
                 params_bounds=distributions['univariate_gmm']['bounds'],
                 initial_parameters=list(
                     chain.from_iterable((i * local_max / no_models, (10 * i + 1), 1 / no_models) for i in range(
                         no_models)))
                 ),
            # 0
            dict(proba=distributions['univariate_gmm']['proba'],
                 params_bounds=distributions['univariate_gmm']['bounds'],
                 initial_parameters=list(
                     chain.from_iterable((i * local_max / no_models, 10 * no_models, 1 / no_models) for i in range(
                         no_models)))
                 ),
        )
        self.gc = None

    def initialize(self, unique_kmers):
        arr = np.vstack([np.array(list(x)) for x in unique_kmers])
        self.gc = np.sum((arr == 'G') | (arr == 'C'), axis=1)

    def compare_kmers(self, i1, i2):
        a = self.gc[i1]
        b = self.gc[i2]
        return np.abs(a - b)


class ShapeMetric(Metric):
    def __init__(self, shape_parameter, no_models):
        super().__init__("shape",
                         "distance",
                         "MSE of chosen shape params")
        self.shape_parameter = shape_parameter
        # TODO set params - should be poisson/exponentials or sth like that
        local_max = 1e3
        super().define_optimalization_params(
            # 1
            dict(proba=distributions['univariate_gmm']['proba'],
                 params_bounds=distributions['univariate_gmm']['bounds'],
                 initial_parameters=list(
                     chain.from_iterable((i * local_max / no_models, (10 * i + 1), 1 / no_models) for i in range(
                         no_models)))
                 ),
            # 0
            dict(proba=distributions['univariate_gmm']['proba'],
                 params_bounds=distributions['univariate_gmm']['bounds'],
                 initial_parameters=list(
                     chain.from_iterable((i * local_max / no_models, 10 * no_models, 1 / no_models) for i in range(
                         no_models)))
                 ),
        )
        self.shape_values = None

    def initialize(self, unique_kmers):
        # with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        robjects.r('library(DNAshapeR)')
        temp_fasta_name = "tmp.fasta"
        with open(temp_fasta_name, mode='w') as temp_fasta:
            # put stuff to tempfile
            for i, item in enumerate(unique_kmers):
                print(f">{i}\n{item}", file=temp_fasta)

            # Call the getShape function
        result = robjects.r('getShape')(temp_fasta_name)

        def strip_nan(array):
            good = np.where(~np.isnan(array[0, :]))[0]
            return array[:, good]

        pyresult = {name: np.array(val) for name, val in zip(result.names, list(result))}
        self.shape_values = strip_nan(pyresult[self.shape_parameter])
        os.remove(temp_fasta_name)
        for name in pyresult.keys():
            os.remove(temp_fasta_name + f".{name}")

    def compare_kmers(self, i1, i2):
        one, two = self.shape_values[i1, :], self.shape_values[i2, :]
        diff = (one - two)
        return np.mean(diff * diff)


class PFMmetric(Metric):
    def __init__(self, name, type, desc,
                 binding_matrix_file,
                 bg_kmers=None, selected_motifs=None,
                 ):
        super().__init__(name, type, desc)
        self.bg_kmers = bg_kmers  # if None then DO NOT NORMALIZE
        self.pfm_file = binding_matrix_file
        self.selected_motifs = selected_motifs
        self.affinities = None

    def load_pfm_database(self, pseudocounts=0.01):
        pfm_database = {}  # motif : pfm
        with open(self.pfm_file) as handle:
            for m in bmotifs.parse(handle, "pfm-four-columns"):
                mname = m.name.split("_")[0]
                m.pseudocounts = pseudocounts
                pfm_database[mname] = m.pssm
        return pfm_database

    def filter_db(self, pfm_database):
        selected_db = {k: pfm_database[k] for k in self.selected_motifs if k in pfm_database}
        return selected_db

    def initialize(self, unique_kmers):
        pfm_database = self.load_pfm_database()
        if self.selected_motifs is not None:
            pfm_database = self.filter_db(pfm_database)
        if self.bg_kmers is not None:
            self.bg_kmers = self.bg_kmers.astype(unique_kmers.dtype)
        results = []

        counter = 0

        for motif in tqdm(pfm_database):
            def score_motif(kmer):
                return score(kmer, pfm_database[motif])

            if pfm_database[motif].length > len(unique_kmers[0]):
                print(f"Skipping {motif} as its longer than k")
                continue

            vectorized_func = np.vectorize(score_motif)
            affinities = vectorized_func(unique_kmers)

            if len(np.where(np.isnan(affinities))[0]):
                raise AttributeError(f"NAN ENCOUNTERED AT {motif}, {np.where(np.isnan(affinities))}")
            if self.bg_kmers is not None:
                bg_affinities = vectorized_func(self.bg_kmers)
                affinities = normalize(affinities, bg_affinities)

            results.append(affinities)
        self.affinities = np.vstack(results).T


class HocomocoIOU(PFMmetric):
    def __init__(self, binding_matrix_file, binder_threshold, no_models, bg_kmers=None, selected_motifs=None):
        super().__init__(
            "hocomoco_iou",
            "similarity",
            "Intersection over union ratio between possible binders.q", binding_matrix_file,
            bg_kmers=bg_kmers,
            selected_motifs=selected_motifs
        )
        self.binder_threshold = binder_threshold
        # TODO optim params setting
        local_max=1e6
        super().define_optimalization_params(
            # 1
            # 1
            dict(proba=distributions['univariate_gmm']['proba'],
                 params_bounds=distributions['univariate_gmm']['bounds'],
                 initial_parameters=list(
                     chain.from_iterable((i * local_max / no_models, (10 * i + 1), 1 / no_models) for i in range(
                         no_models)))
                 ),
            # 0
            dict(proba=distributions['univariate_gmm']['proba'],
                 params_bounds=distributions['univariate_gmm']['bounds'],
                 initial_parameters=list(
                     chain.from_iterable((i * local_max / no_models, 10 * no_models, 1 / no_models) for i in range(
                         no_models)))
                 ),
        )

    def initialize(self, unique_kmers):
        super().initialize(unique_kmers)
        self.affinities = self.affinities >= self.binder_threshold

    def compare_kmers(self, i1, i2):
        one, two = self.affinities[i1, :], self.affinities[i2, :]

        intersect = np.sum(one & two)
        union = np.sum(one | two)
        if union == 0:
            return 0
        return intersect / union


class HocomocoMSE(PFMmetric):
    def __init__(self, binding_matrix_file,  no_models,  # binder_threshold,
                 bg_kmers=None, selected_motifs=None):
        super().__init__(
            "hocomoco_mse",
            "distance",
            "MSE with HOCOMOCO affinities", binding_matrix_file,
            bg_kmers=bg_kmers,
            selected_motifs=selected_motifs
        )
        # self.binder_threshold = binder_threshold
        local_max=1e8
        super().define_optimalization_params(
            # 1
            dict(proba=distributions['univariate_gmm']['proba'],
                 params_bounds=distributions['univariate_gmm']['bounds'],
                 initial_parameters=list(
                     chain.from_iterable((i * local_max / no_models, (10 * i + 1), 1 / no_models) for i in range(
                         no_models)))
                 ),
            # 0
            dict(proba=distributions['univariate_gmm']['proba'],
                 params_bounds=distributions['univariate_gmm']['bounds'],
                 initial_parameters=list(
                     chain.from_iterable((i * local_max / no_models, 10 * no_models, 1 / no_models) for i in range(
                         no_models)))
                 ),
        )

    def initialize(self, unique_kmers):
        super().initialize(unique_kmers)

    def compare_kmers(self, i1, i2):
        one, two = self.affinities[i1, :], self.affinities[i2, :]
        diff = (one - two)
        return np.mean(diff * diff)


class ProBoundMetric(Metric):
    def __init__(self, name, type, desc,
                 bg_kmers=None, selected_motifs=None,
                 taxa=None  # list of strings -- taxa in MotifCentral
                 ):
        super().__init__(name, type, desc)
        self.bg_kmers = bg_kmers  # if None then DO NOT NORMALIZE
        self.selected_motifs = selected_motifs
        self.affinities = None
        self.taxa = taxa

    def filter_db(self, models, names):
        mask = names.isin(self.selected_motifs)
        return models[mask], names[mask]

    def initialize(self, unique_kmers):
        mc = pyProBound.MotifCentral()
        if self.taxa is not None:
            mc = mc.filter(taxa=self.taxa)
        mc = mc[mc["gene_symbols"].str.len() == 1]
        models, names = mc['model_id'].values.astype(int), mc["gene_symbols"].str[0].values

        if self.selected_motifs is not None:
            models, names = self.filter_db(models, names)

        simplified_kmers = [str(x) for x in unique_kmers]
        if self.bg_kmers is not None:
            self.bg_kmers = [str(x) for x in self.bg_kmers]
        # for data compatibility with Java, pyjnius had some issues with numpy and its data types

        results = []
        counter = 0

        for model_id, name in tqdm(list(zip(models, names))):
            model_file = os.path.join(script_dir, "sources_for_motif_based_scoring", "probound", f"{model_id}.json")
            model = pyProBound.ProBoundModel(model_file, fitjson=True)

            model.select_binding_mode(0)  # in the motif central models there is usually only one

            affinities = model.score_affinity_sum(simplified_kmers)
            if self.bg_kmers is not None:
                bg_affinities = model.score_affinity_sum(self.bg_kmers)
                affinities = normalize(affinities, bg_affinities)

            # if counter > 4:
            #     break
            # counter += 1

            results.append(np.array(affinities))

        self.affinities = np.vstack(results)


class ProBoundHumanMSE(ProBoundMetric):
    def __init__(self, no_models, bg_kmers=None, selected_motifs=None,
                 ):
        super().__init__(
            "probound_mse",
            "distance",
            "Affinity (z-score) ",
            taxa=["Homo sapiens"],
            bg_kmers=bg_kmers,
            selected_motifs=selected_motifs
        )
        # TODO optim params setting
        local_max=1e8
        super().define_optimalization_params(
            # 1
            dict(proba=distributions['univariate_gmm']['proba'],
                 params_bounds=distributions['univariate_gmm']['bounds'],
                 initial_parameters=list(
                     chain.from_iterable((i * local_max / no_models, (10 * i + 1), 1 / no_models) for i in range(
                         no_models)))
                 ),
            # 0
            dict(proba=distributions['univariate_gmm']['proba'],
                 params_bounds=distributions['univariate_gmm']['bounds'],
                 initial_parameters=list(
                     chain.from_iterable((i * local_max / no_models, 10 * no_models, 1 / no_models) for i in range(
                         no_models)))
                 ),
        )

    def compare_kmers(self, i1, i2):
        one, two = self.affinities[i1, :], self.affinities[i2, :]
        diff = (one - two)
        return np.mean(diff * diff)
