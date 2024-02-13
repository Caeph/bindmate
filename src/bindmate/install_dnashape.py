import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages

# Install and load the necessary packages
if not rpackages.isinstalled('BiocManager'):
    print("INSTALLING BIOMANAGER")
    robjects.r('install.packages("BiocManager", INSTALL_opts=c("--no-lock", "--no-plots", "--no-html", "--no-data", "--no-help", "--no-docs",  "--no-demo", "--no-exec", "--no-byte-compile", "--no-test-load"))'),

robjects.r('library(BiocManager)')
# Install DNAshapeR from Bioconductor
robjects.r('BiocManager::install("DNAshapeR")')