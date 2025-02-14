import pytest
import numpy as np
from polymetrix.polymer import Polymer
from polymetrix.featurizer import StarToSidechainMinDistanceFeaturizer


def test_backbone_to_sidechain_distance_simple():
    """Test basic polymer with single side chain."""
    polymer = Polymer.from_psmiles("*C(C=C)C*")
    featurizer = StarToSidechainMinDistanceFeaturizer(agg=["mean", "min", "max"])
    features = featurizer.featurize(polymer)
    labels = featurizer.feature_labels()
    assert len(features) == 3
    assert all(isinstance(f, float) for f in features)
    
    
def test_backbone_to_sidechain_distance_multiple_chains():
    """Test polymer with multiple side chains."""
    polymer = Polymer.from_psmiles("*CC(CC(*)(C#N)C#N)c1ccc(CCl)cc1")
    featurizer = StarToSidechainMinDistanceFeaturizer(agg=["mean", "min", "max"])
    features = featurizer.featurize(polymer)
    assert len(features) == 3
    # Multiple side chains should result in different min/max values
    assert features[1] != features[2]  # min != max
    
def test_backbone_to_sidechain_distance_with_ring():
    """Test polymer with ring structure in side chain."""
    polymer = Polymer.from_psmiles("*CC(c1ccccc1)C*")
    featurizer = StarToSidechainMinDistanceFeaturizer(agg=["mean", "min", "max"])
    features = featurizer.featurize(polymer)
    assert len(features) == 3
    # Ring structure should have consistent non-zero distances
    assert all(f > 0 for f in features)
    
def test_backbone_to_sidechain_distance_branched():
    """Test polymer with branched side chains."""
    polymer = Polymer.from_psmiles("*Nc1ccc(NC(=O)c2ccc(C(c3ccc(C(*)=O)c(C(=O)OCC)c3)(C(F)(F)F)C(F)(F)F)cc2C(=O)OCC)cc1")
    featurizer = StarToSidechainMinDistanceFeaturizer(agg=["mean", "min", "max"])
    features = featurizer.featurize(polymer)
    assert len(features) == 3
    # Branched structure should have larger max distance
    assert features[2] > features[0]  # max > mean
    

def test_backbone_to_sidechain_distance_edge_cases():
    """Test edge cases with minimal and complex structures."""
    # Minimal case - single atom side chain
    polymer_minimal = Polymer.from_psmiles("*C(C)C*")
    # Complex case - multiple features
    polymer_complex = Polymer.from_psmiles("*Nc1ccc(NC(=O)c2ccc(C(c3ccc(C(*)=O)c(C(=O)OCC)c3)(C(F)(F)F)C(F)(F)F)cc2C(=O)OCC)cc1")

    featurizer = StarToSidechainMinDistanceFeaturizer(agg=["mean", "min", "max"])

    features_minimal = featurizer.featurize(polymer_minimal)
    features_complex = featurizer.featurize(polymer_complex)

    assert len(features_minimal) == 3
    assert len(features_complex) == 3
    # Complex structure should have larger distances
    assert features_complex[2] > features_minimal[2]  # max distance comparison
