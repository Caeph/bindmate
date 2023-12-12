# imports here
from kmer_to_kmer_matchscores import calculate_kmer_to_kmer_matchscores, PairingResults
import input_loading
import os
from seq_to_seq_matchscores import *


class PairingProbabilityCalculator:
    def __init__(self, k, metrics, material_saving_dir,
                 max_em_step=30, preselection_part=0.5,
                 background_type=None, background_size=None, **bgkwargs):
        # set up kmer similarity functions
        # set up preselection no
        self.k = k
        self.metrics = metrics
        self.material_saving_dir = material_saving_dir
        self.background = [background_type, background_size, bgkwargs]
        self.max_em_step = max_em_step
        self.preselection_part = preselection_part

    def add_user_defined_metric(self, function):
        # todo fill out missing stuff
        ...

    def __write_descriptions(self, input):
        os.makedirs(self.material_saving_dir, exist_ok=True)
        descriptions = [
            (self.k, "k"),
            (self.background, "background"),
            (self.metrics, "metrics"),
            (input, "input")
        ]
        with open(os.path.join(self.material_saving_dir, "readme.txt"), mode='w') as writer:
            for value, desc in descriptions:
                print(f"{desc}: {value}", file=writer)

    def fit_predict_fasta(self, fasta_filename):
        # read fasta and prep input
        # analyze
        self.__write_descriptions(fasta_filename)
        sequence_df = input_loading.load_fasta_input(fasta_filename)
        return self.fit(sequence_df, to_file=True)

    def fit_predict_bed(self, bed_filename, source_fasta):
        # read bed and prep input
        # analyze
        self.__write_descriptions([bed_filename, source_fasta])
        sequence_df = input_loading.load_bed_input(bed_filename, source_fasta)
        similarities = self.fit(sequence_df, to_file=True)
        return similarities

    def fit(self, sequences, to_file=None):

        # return fit_result
        optimized = calculate_kmer_to_kmer_matchscores(sequences, self.k, self.metrics,
                                                       self.background,
                                                       save_results=os.path.join(
                                                        self.material_saving_dir, "metric_ranks.csv"
                                                       ), em_params_file=os.path.join(
                                                        self.material_saving_dir, "recorded_parameters.csv"
                                                       ),
                                                       max_em_step=self.max_em_step,
                                                       preselection_part=self.preselection_part
                                                       )
        # save stuff
        optimized.save(self.material_saving_dir)
        seq2seq = calculate_seq_to_seq_similarities(optimized)
        if to_file:
            with open(os.path.join(self.material_saving_dir, "seq2seq_results.csv"),
                      mode='w') as writer:
                for item in seq2seq:
                    print(item.to_string(), file=writer)
        return seq2seq


class AnalyticalTool:
    def __init__(self, pairing_result, seq2seq_result, info_ranks):
        self.pairing_result = pairing_result
        self.seq2seq = seq2seq_result
        self.info_ranks = info_ranks

    @staticmethod
    def load(dirname):
        pairing_result = PairingResults.load(dirname)
        seq2seq = []
        for line in open(os.path.join(dirname, "seq2seq_results.csv")):
            line = line.strip("\n")
            item = SeqToSeqPairing.load(line)
            seq2seq.append(item)

        info_ranks = pd.read_csv(os.path.join(dirname, "metric_ranks.csv"))
        return AnalyticalTool(pairing_result, seq2seq, info_ranks)
