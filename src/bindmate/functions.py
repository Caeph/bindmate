# imports here
from predefined_functions import predefined_functions as __predefined_functions
from kmer_to_kmer_matchscores import calculate_kmer_to_kmer_matchscores
import input_loading

#TODO THIS IS A PLANNED FUNCTIONALITY

# optim info -- dictionary
# { 
#       probability_func(value, parameters),
#       argmax_func(q, x)
# }


class PairingProbabilityCalculator:
    def __init__(self):
        # set up kmer similarity functions
        # set up preselection no
        self.k = ...
        self.metrics = ...
        ...

    def add_user_defined_metric(self, function):
        # todo check if is symmetric
        # todo fill out missing stuff
        ...

    def fit_predict_fasta(self, fasta_filename):
        # read fasta and prep input
        # analyze
        sequence_df = input_loading.load_fasta_input(fasta_filename)
        self.fit(sequence_df)

    def fit_predict_bed(self, bed_filename, source_fasta):
        # read bed and prep input
        # analyze
        sequence_df = input_loading.load_bed_input(bed_filename, source_fasta)
        self.fit(sequence_df)

    def fit(self, sequences):
        # create a list of unique kmers
        # calculate kmer similarity/distance functions
        # calculate match probabilities
        # preselection filtering
        # best pairing in sequences

        # return fit_result
        optimized = calculate_kmer_to_kmer_matchscores(sequences, self.k, self.metrics)

    def analyze(self, fit_result):
        ...


# TODO analytic tools
def supervised_analytics(metric, metching_pairs, mismatched_pairs):
    ...


# TODO viewing methods for fitted and analysed stuff
def view():
    ...
