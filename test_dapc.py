# Standard library imports
import argparse
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

# Custom module imports
from read_input.read_input import GenotypeData
from read_input.impute import *

from sklearn_genetic.space import Continuous, Categorical, Integer

from dim_reduction.dim_reduction import DimReduction
from dim_reduction.embed import *

from clustering.clustering import *

from discriminant_analysis.discriminant_analysis import DiscriminantAnalysis

def main():
	"""[Class instantiations and main package body]
	"""

	args = get_arguments()

	data = GenotypeData(
		filename="example_data/phylip_files/test2.phy", 
		filetype="phylip", 
		popmapfile="example_data/popmaps/test2.popmap"
	)

	imp = ImputeAlleleFreq(
		genotype_data = data,
		pops=data.pops
	)
	print(imp.imputed)
	#pca.coords dimensions: samples X pcs
	pca = runPCA(
		gt=imp.imputed, 
		pops=data.pops, 
		sampleids=data.samples,
		prefix="test", 
		scaler="patterson",
		plot_cumvar=False, 
		keep_pcs=100
	)
	pca.plot()
	
	dapc = DiscriminantAnalysis(
		embedding = pca,
		pops = data.pops,
		prefix="test",
		solver="svd",
		run_xval=True,
		n_pca_min=10,
		n_pca_max=160,
		n_pca_iter=20,
		xval_reps=10
	)
	dapc.plot(axis1=1, axis2=2)
	dapc.plot(axis1=2, axis2=3)
	
	
def get_arguments():
	"""[Parse command-line arguments. Imported with argparse]

	Returns:
		[argparse object]: [contains command-line arguments; accessed as method]
	"""

	parser = argparse.ArgumentParser(description="Convert VCF file to BGC format (with genotype uncertainties). Currently only handles three populations maximum (P1, P2, and Admixed).", add_help=False)

	required_args = parser.add_argument_group("Required arguments")
	filetype_args = parser.add_argument_group("File type arguments (choose only one)")
	structure_args = parser.add_argument_group("Structure file arguments")
	optional_args = parser.add_argument_group("Optional arguments")

	# File Type arguments
	filetype_args.add_argument("-s", "--str",
								type=str,
								required=False,
								help="Input structure file")
	filetype_args.add_argument("-p", "--phylip",
								type=str,
								required=False,
								help="Input phylip file")
	
	filetype_args.add_argument("-t", "--treefile",
								type=str,
								required=False,
								default=None,
								help="Newick-formatted treefile")

	# Structure Arguments
	structure_args.add_argument("--onerow_perind",
								default=False,
								action="store_true",
								help="Toggles on one row per individual option in structure file")
	structure_args.add_argument("--pop_ids",
								default=False,
								required=False,
								action="store_true",
								help="Toggles on population ID column (2nd col) in structure file")
	
	
	## Optional Arguments
	optional_args.add_argument("-m", "--popmap",
								type=str,
								required=False,
								default=None,
								help="Two-column tab-separated population map file: inds\tpops. No header line")
	optional_args.add_argument("--prefix",
								type=str,
								required=False,
								default="output",
								help="Prefix for output files")

	optional_args.add_argument("--resume_imputed",
								type=str,
								required=False,
								help="Read in imputed data from a file instead of doing the imputation")						
	# Add help menu							
	optional_args.add_argument("-h", "--help",
								action="help",
								help="Displays this help menu")

	# If no command-line arguments are called then exit and call help menu.
	if len(sys.argv)==1:
		print("\nExiting because no command-line options were called.\n")
		parser.print_help(sys.stderr)
		sys.exit(1)

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	main()
