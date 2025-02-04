import pytest
import numpy as np
from polymetrix.polymer import Polymer
from polymetrix.featurizer import SidechainToBackboneRatioFeaturizer

def test_sidechain_to_backbone_ratio():
    polymer = Polymer.from_psmiles("*CC(CC(*)(C#N)C#N)c1ccc(CCl)cc1")
    featurizer = SidechainToBackboneRatioFeaturizer(agg=["mean", "min", "max"])
    features = featurizer.featurize(polymer)
    labels = featurizer.feature_labels()
    
    assert len(features) == 3 
    assert len(labels) == 3
    expected_values = np.array([1.56, 1.00, 2.67])
    np.testing.assert_array_almost_equal(features, expected_values, decimal=2)

def test_empty_polymer():
    polymer = Polymer.from_psmiles("*CC*")
    featurizer = SidechainToBackboneRatioFeaturizer(agg=["mean", "min", "max"])
    
    features = featurizer.featurize(polymer)
    assert np.all(features == 0)
    assert len(features) == 3

def test_single_aggregation():
    polymer = Polymer.from_psmiles("*CC(CC(*)(C#N)C#N)c1ccc(CCl)cc1")
    featurizer = SidechainToBackboneRatioFeaturizer(agg=["mean"])
    
    features = featurizer.featurize(polymer)
    labels = featurizer.feature_labels()
    
    assert len(features) == 1
    assert len(labels) == 1
    np.testing.assert_array_almost_equal(features, [1.56], decimal=2)

def test_invalid_polymer():
    with pytest.raises(Exception):
        polymer = Polymer.from_psmiles("invalid_smiles")
        featurizer = SidechainToBackboneRatioFeaturizer()
        featurizer.featurize(polymer)