# udapc/analysis/base.py
from abc import ABC, abstractmethod

class DiscriminantAnalysis(ABC):
    def __init__(self, genotype_data, pops=None, sample_ids=None, prefix=None):
        self.genotype_data = genotype_data
        self.pops = pops
        self.sample_ids = sample_ids
        self.prefix = prefix

    @abstractmethod
    def fit_predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def score(self, *args, **kwargs):
        pass
