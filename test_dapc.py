# Standard library imports
import argparse
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

# Custom module imports
import udapc
from udapc.read_input.read_input import GenotypeData
from udapc.simple_imputers.simple_imputers import ImputeAlleleFreq
from udapc.dim_reduction.embed import runPCA 
from udapc.discriminant_analysis.discriminant_analysis import * 

# from dim_reduction.dim_reduction import DimReduction
# from dim_reduction.embed import *

# from clustering.clustering import *

# from discriminant_analysis.discriminant_analysis import DiscriminantAnalysis

def main():
    """[Class instantiations and main package body]
    """

    data = GenotypeData(
        filename="example_data/mah_sub100_subset.recode.vcf.gz", 
        #filename="example_data/mah_thin.vcf.gz", 
        #filename="example_data/mah_thin_subset.recode.vcf.gz",
        filetype="vcf", 
        popmapfile="example_data/mah_drainage.popmap"
    )
    print(data.genotypes012_array)

    imp = ImputeAlleleFreq(
        genotype_data = data,
        by_populations=True
    )
    print(imp.imputed.genotypes012_array)

    #pca.coords dimensions: samples X pcs
    pca = runPCA(
        gt=imp.imputed.genotypes012_array, 
        pops=data.pops, 
        sampleids=data.samples,
        prefix="test", 
        scaler="patterson",
        plot_cumvar=False
    )
    pca.plot()
    
    # dapc = DiscriminantAnalysis(
    #     embedding = pca,
    #     pops = data.pops,
    #     prefix="test",
    #     solver="svd",
    #     run_xval=True,
    #     n_pca_min=160,
    #     n_pca_max=400,
    #     n_pca_iter=20,
    #     xval_reps=20
    # )
    # dapc.plot(axis1=1, axis2=2)
    # dapc.plot(axis1=2, axis2=3)

    udapc = UnsupervisedDiscriminantAnalysis(
        embedding = pca, 
        pops = data.pops,
        n_pca_min=10,
        n_pca_max=160,
        n_pca_iter=20,
        reps = 20, 
        n_init = 20, 
        n_try = 50, 
        max_iter = 1000, 
        maxk = 8
    )
    


if __name__ == "__main__":
    main()
