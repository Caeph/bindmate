import rpy2.robjects as robjects
import numpy as np
# ChatGPT generated

# Install and load the necessary packages
# if not rpackages.isinstalled('BiocManager'):
#     robjects.r('install.packages("BiocManager")')
# robjects.r('library(BiocManager)')
# # Install DNAshapeR from Bioconductor
# robjects.r('BiocManager::install("DNAshapeR")')

# Load the DNAshapeR package
robjects.r('library(DNAshapeR)')

# Define a DNA sequence (replace 'your_dna_sequence' with your actual DNA sequence)
dna_sequence = "AAAAAAACTGCTGCTGCTGCAAAAAAAAAAAAAAAAAA"

unique_kmers = np.array([dna_sequence])
# with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_fasta:
temp_fasta_name = "tmp.fasta"
with open(temp_fasta_name, mode='w') as temp_fasta:
    # put stuff to tempfile
    for i, item in enumerate(unique_kmers):
        print(f">{i}\n{item}", file=temp_fasta)

    # Call the getShape function
result = robjects.r('getShape')(temp_fasta_name)
pyresult = {name:np.array(val) for name,val in zip(result.names, list(result))}
print(pyresult)

