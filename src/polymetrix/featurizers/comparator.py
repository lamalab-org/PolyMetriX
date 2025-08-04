import numpy as np
from typing import ClassVar


class PolymerMoleculeComparator:
    """Comparator that computes various comparison metrics between polymer and molecule features."""

    comparators: ClassVar = {
        "absolute_difference": lambda p, m: np.abs(p - m),
        "signed_difference": lambda p, m: p - m,
        "product": lambda p, m: p * m,
        "squared_distance": lambda p, m: (p - m) ** 2,
        "euclidean_distance": lambda p, m: np.sqrt((p - m) ** 2),
    }

    agg_funcs: ClassVar = {
        "mean": np.mean,
        "min": np.min,
        "max": np.max,
        "sum": np.sum,
    }

    def __init__(
        self, polymer_featurizer, molecule_featurizer, comparisons=None, agg=None
    ):
        self.polymer_featurizer = polymer_featurizer
        self.molecule_featurizer = molecule_featurizer
        self.comparisons = (
            comparisons if comparisons is not None else ["absolute_difference"]
        )
        self.agg = agg if agg is not None else []

    def _concatenate_results(self, results):
        return np.concatenate(results) if results else np.array([])

    def compare(self, polymer, molecule):
        """Return comparison metrics between polymer and molecule features."""
        polymer_features = self.polymer_featurizer.featurize(polymer).flatten()
        molecule_features = self.molecule_featurizer.featurize(molecule).flatten()

        results = []
        for comp_func in self.comparisons:
            if comp_func not in self.comparators:
                raise ValueError(f"Unknown comparison function: {comp_func}")
            comparison_result = self.comparators[comp_func](
                polymer_features, molecule_features
            )
            results.append(comparison_result)

        if self.agg:
            aggregated_results = self.aggregate(results)
            results.append(aggregated_results)

        return self._concatenate_results(results)

    def aggregate(self, features):
        """Aggregate features across comparison methods."""
        results = []
        for agg_func in self.agg:
            if agg_func not in self.agg_funcs:
                raise ValueError(f"Unknown aggregation function: {agg_func}")
            aggregated = self.agg_funcs[agg_func](features, axis=0)
            results.append(aggregated)

        return self._concatenate_results(results)

    def _generate_labels(self, base_labels, suffix):
        return [f"{label}_{suffix}" for label in base_labels]

    def feature_labels(self):
        """Generate labels for comparison and aggregated features."""
        base_labels = self.polymer_featurizer.feature_labels()
        labels = []

        # Labels for comparison functions
        for comp_func in self.comparisons:
            labels.extend(self._generate_labels(base_labels, comp_func))

        # Labels for aggregated results
        if self.agg:
            comparison_methods_str = "_".join(self.comparisons)
            for agg_func in self.agg:
                labels.extend(
                    self._generate_labels(
                        base_labels, f"{comparison_methods_str}_{agg_func}"
                    )
                )

        return labels
