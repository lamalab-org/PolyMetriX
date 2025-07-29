import numpy as np


class PolymerMoleculeComparator:
    """comparator that computes absolute difference between polymer and molecule features."""

    def __init__(self, polymer_featurizer, molecule_featurizer):
        self.polymer_featurizer = polymer_featurizer
        self.molecule_featurizer = molecule_featurizer

    def compare(self, polymer, molecule):
        """Return absolute difference between polymer and molecule features."""
        polymer_features = self.polymer_featurizer.featurize(polymer).flatten()
        molecule_features = self.molecule_featurizer.featurize(molecule).flatten()

        return np.abs(polymer_features - molecule_features)

    def feature_labels(self):
        """Return feature labels with '_difference' suffix."""
        labels = self.polymer_featurizer.feature_labels()
        return [f"{label}_difference" for label in labels]
