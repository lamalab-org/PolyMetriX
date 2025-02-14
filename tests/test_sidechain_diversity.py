import pytest
import numpy as np
from polymetrix.polymer import Polymer
from polymetrix.featurizer import SidechainDiversityFeaturizer

def test_sidechain_diversity_no_sidechains():
    """Test polymer with no sidechains."""
    polymer = Polymer.from_psmiles("*CC*") 
    featurizer = SidechainDiversityFeaturizer()
    features = featurizer.featurize(polymer)
    assert features == np.array([0])
    assert featurizer.feature_labels() == ["num_diverse_sidechains"]

def test_sidechain_diversity_complex_branching():
    polymer = Polymer.from_psmiles("*C(=O)Nc1c(CC)cc(Cc2cc(CC)c(NC(=O)c3ccc4c(c3)C(=O)N(c3ccc(/N=N/c5ccc(C)cc5)c(N5C(=O)c6ccc(*)cc6C5=O)c3)C4=O)c(CC)c2)cc1CC" )
    featurizer = SidechainDiversityFeaturizer()
    features = featurizer.featurize(polymer)
    assert features == np.array([2])  
