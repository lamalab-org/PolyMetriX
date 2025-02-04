import pytest
import numpy as np
from polymetrix.polymer import Polymer
from polymetrix.featurizer import BackboneToSidechainDistanceFeaturizer

def test_backbone_to_sidechain_distance():
    polymer = Polymer.from_psmiles("*C(C=C)C*")
    featurizer = BackboneToSidechainDistanceFeaturizer(agg=["mean", "min", "max"])
    
 
    features = featurizer.featurize(polymer)
    labels = featurizer.feature_labels()
    
 
    assert len(features) == 3  
    assert len(labels) == 3
    
    expected_values = np.array([1.00, 1.00, 1.00])
    np.testing.assert_array_almost_equal(features, expected_values, decimal=2)
    

def test_no_sidechain():
    polymer = Polymer.from_psmiles("*CC*")
    featurizer = BackboneToSidechainDistanceFeaturizer(agg=["mean", "min", "max"])
    
    features = featurizer.featurize(polymer)
    
    assert np.all(features == 0)
    assert len(features) == 3

def test_single_aggregation():
    polymer = Polymer.from_psmiles("*C(C=C)C*")
    featurizer = BackboneToSidechainDistanceFeaturizer(agg=["mean"])
    
    features = featurizer.featurize(polymer)
    labels = featurizer.feature_labels()
    
    assert len(features) == 1
    assert len(labels) == 1
    np.testing.assert_array_almost_equal(features, [1.00], decimal=2)


def test_invalid_polymer():
    with pytest.raises(Exception):
        polymer = Polymer.from_psmiles("invalid_smiles")
        featurizer = BackboneToSidechainDistanceFeaturizer()
        featurizer.featurize(polymer)

