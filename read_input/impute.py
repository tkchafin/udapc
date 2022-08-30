# Standard library imports
import sys
import os
from collections import Counter
from operator import itemgetter
from statistics import mean
from pathlib import Path

# Third party imports
import numpy as np
import pandas as pd

from utils.misc import get_processor_name

# Requires scikit-learn-intellex package
if get_processor_name().strip().startswith("Intel"):
	try:
		from sklearnex import patch_sklearn
		patch_sklearn()
		intelex = True
	except ImportError:
		print("Warning: Intel CPU detected but scikit-learn-intelex is not installed. We recommend installing it to speed up computation.")
		intelex = False
else:
	intelex = False


# Custom module imports
from read_input.read_input import GenotypeData
from utils import misc
from utils.misc import timer
#from utils.misc import bayes_search_CV_init
from utils import sequence_tools

from utils.misc import isnotebook

is_notebook = isnotebook()

if is_notebook:
	from tqdm.notebook import tqdm as progressbar
else:
	from tqdm import tqdm as progressbar

class ImputeAlleleFreq(GenotypeData):
	"""[Class to impute missing data by global or population-wisse allele frequency]

	Args:
		GenotypeData ([GenotypeData]): [Inherits from GenotypeData class that reads input data from a sequence file]
	"""
	def __init__(
		self,
		genotype_data,
		pops=None,
		diploid=True,
		default=0,
		missing=-9
	):
		"""[Impute missing data by global allele frequency. Population IDs can be sepcified with the pops argument. if pops is None, then imputation is by global allele frequency. If pops is not None, then imputation is by population-wise allele frequency. A list of population IDs in the appropriate format can be obtained from the GenotypeData object as GenotypeData.populations]

		Args:
			pops ([list(str)], optional): [If None, then imputes by global allele frequency. If not None, then imputes population-wise and pops should be a list of population assignments. The list of population assignments can be obtained from the GenotypeData object as GenotypeData.populations]. Defaults to None.

			diploid (bool, optional): [When diploid=True, function assumes 0=homozygous ref; 1=heterozygous; 2=homozygous alt. 0-1-2 genotypes are decomposed to compute p (=frequency of ref) and q (=frequency of alt). In this case, p and q alleles are sampled to generate either 0 (hom-p), 1 (het), or 2 (hom-q) genotypes. When diploid=FALSE, 0-1-2 are sampled according to their observed frequency]. Defaults to True.

			default (int, optional): [Value to set if no alleles sampled at a locus]. Defaults to 0.

			missing (int, optional): [Missing data value]. Defaults to -9.
		"""
		super().__init__()

		self.pops = pops
		self.diploid = diploid
		self.default = default
		self.missing = missing

		self.imputed = self.fit_predict(genotype_data.genotypes_list)

	@timer
	def fit_predict(self, X):
		"""[Impute missing genotypes using allele frequencies, with missing alleles coded as negative; usually -9]

		Args:
			X ([list(list(int))]): [012-encoded genotypes obtained from the GenotypeData object as GenotypeData.genotypes_list]

		Returns:
			[pandas.DataFrame]: [Imputed genotypes of same dimensions as data]
		"""
		if self.pops:
			print("\nImputing by population allele frequencies...")
		else:
			print("\nImputing by global allele frequency...")

		data = [item[:] for item in X]

		if self.pops is not None:
			pop_indices = misc.get_indices(self.pops)

		loc_index=0
		for locus in data:
			if self.pops is None:
				allele_probs = self._get_allele_probs(locus, self.diploid)
				#print(allele_probs)
				if misc.all_zero(list(allele_probs.values())) or \
					not allele_probs:
					print("\nWarning: No alleles sampled at locus",
						str(loc_index),
						"setting all values to:",str(self.default))
					gen_index=0
					for geno in locus:
						data[loc_index][gen_index] = self.default
						gen_index+=1

				else:
					gen_index=0
					for geno in locus:
						if geno == self.missing:
							data[loc_index][gen_index] = \
								self._sample_allele(allele_probs, diploid=True)
						gen_index += 1

			else:
				for pop in pop_indices.keys():
					allele_probs = self._get_allele_probs(
						locus, self.diploid,
						missing=self.missing,
						indices=pop_indices[pop]
					)

					if misc.all_zero(list(allele_probs.values())) or not allele_probs:
						print("\nWarning: No alleles sampled at locus",
							str(loc_index),
							"setting all values to:",
							str(self.default)
						)
						gen_index=0
						for geno in locus:
							data[loc_index][gen_index] = self.default
							gen_index += 1
					else:
						gen_index=0
						for geno in locus:
							if geno == self.missing:
								data[loc_index][gen_index] = \
									self._sample_allele(
										allele_probs,
										diploid=True
									)
							gen_index += 1

			loc_index += 1

		df = pd.DataFrame(data)

		print("Done!")
		return df

	def _sample_allele(self, allele_probs, diploid=True):
		if diploid:
			alleles=misc.weighted_draw(allele_probs, 2)
			if alleles[0] == alleles[1]:
				return alleles[0]
			else:
				return 1
		else:
			return misc.weighted_draw(allele_probs, 1)[0]

	def _get_allele_probs(
		self, genotypes, diploid=True, missing=-9, indices=None
	):
		data=genotypes
		length=len(genotypes)

		if indices is not None:
			data = [genotypes[index] for index in indices]
			length = len(data)

		if len(set(data))==1:
			if data[0] == missing:
				ret=dict()
				return ret
			else:
				ret=dict()
				ret[data[0]] = 1.0
				return ret

		if diploid:
			length = length*2
			ret = {0:0.0, 2:0.0}
			for g in data:
				if g == 0:
					ret[0] += 2
				elif g == 2:
					ret[2] += 2
				elif g == 1:
					ret[0] += 1
					ret[2] += 1
				elif g == missing:
					length -= 2
				else:
					print("\nWarning: Ignoring unrecognized allele",
						str(g),
						"in get_allele_probs\n"
					)
			for allele in ret.keys():
				ret[allele] = ret[allele] / float(length)
			return ret
		else:
			ret=dict()
			for key in set(data):
				if key != missing:
					ret[key] = 0.0
			for g in data:
				if g == missing:
					length -= 1
				else:
					ret[g] += 1
			for allele in ret.keys():
				ret[allele] = ret[allele] / float(length)
			return ret
