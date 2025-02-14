import pytest
import numpy as np
from polymetrix.polymer import Polymer
from polymetrix.featurizer import SidechainLengthToStarAttachmentDistanceRatioFeaturizer

def test_basic_sidechain_to_backbone_ratio():
    polymer = Polymer.from_psmiles("*CC(CC(*)(C#N)C#N)c1ccc(CCl)cc1")
    featurizer = SidechainLengthToStarAttachmentDistanceRatioFeaturizer(agg=["mean", "min", "max"])
    features = featurizer.featurize(polymer)
    labels = featurizer.feature_labels()
    
    assert len(features) == 3 
    assert len(labels) == 3
    expected_values = np.array([1.56, 1.00, 2.67])
    np.testing.assert_array_almost_equal(features, expected_values, decimal=2)
    
    
def test_ring_in_backbone():
    polymer = Polymer.from_psmiles("*c1ccc(*)cc1")  # Simple phenylene backbone
    featurizer = SidechainLengthToStarAttachmentDistanceRatioFeaturizer(agg=["mean", "min", "max"])
    features = featurizer.featurize(polymer)
    expected_values = np.array([0.0, 0.0, 0.0])  # No side chains
    np.testing.assert_array_almost_equal(features, expected_values, decimal=2)
    
def test_ring_in_sidechain():
    polymer = Polymer.from_psmiles("*CC(c1ccccc1)CC*")  # Phenyl side chain
    featurizer = SidechainLengthToStarAttachmentDistanceRatioFeaturizer(agg=["mean", "min", "max"])
    features = featurizer.featurize(polymer)
    expected_values = np.array([2.0, 2.0, 2.0]) 
    np.testing.assert_array_almost_equal(features, expected_values, decimal=2)
    
def test_long_sidechain():
    polymer = Polymer.from_psmiles("*CC(CCCCCCCCC)CC*")  # Long alkyl side chain
    featurizer = SidechainLengthToStarAttachmentDistanceRatioFeaturizer(agg=["mean", "min", "max"])
    features = featurizer.featurize(polymer)
    expected_values = np.array([3.0, 3.0, 3.0])  
    np.testing.assert_array_almost_equal(features, expected_values, decimal=2)
    
def test_heteroatoms():
    polymer = Polymer.from_psmiles("*C(C#N)C*")  # Side chain with heteroatoms
    featurizer = SidechainLengthToStarAttachmentDistanceRatioFeaturizer(agg=["mean", "min", "max"])
    features = featurizer.featurize(polymer)
    expected_values = np.array([1.0, 1.0, 1.0])  
    np.testing.assert_array_almost_equal(features, expected_values, decimal=2)