from typing import List, Optional
import numpy as np
from rdkit import Chem


class BaseFeatureCalculator:
    agg_funcs = {
        "mean": np.mean,
        "min": np.min,
        "max": np.max,
        "sum": np.sum,
    }

    def __init__(self, agg: List[str] = None):
        if agg is None:
            agg = ["sum"]
        self.agg = agg

    def _sanitize(self, mol: Chem.Mol, sanitize: bool) -> None:
        """Handle molecule sanitization with kekulization exception handling."""
        if sanitize:
            try:
                Chem.SanitizeMol(
                    mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE
                )
            except Chem.AtomKekulizeException:
                mol.UpdatePropertyCache()

    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        raise NotImplementedError("Calculate method must be implemented by subclasses")

    def feature_base_labels(self) -> List[str]:
        raise NotImplementedError(
            "Feature labels method must be implemented by subclasses"
        )

    def feature_labels(self) -> List[str]:
        return [
            f"{label}_{agg}" for label in self.feature_base_labels() for agg in self.agg
        ]

    def aggregate(self, features: List) -> np.ndarray:
        """
        Aggregates a list of features using the aggregation functions specified in self.agg.
        If the features are numpy arrays, the aggregation is applied along the first axis.
        Otherwise, the aggregation is applied directly (assuming the features are scalar numeric values).
        """
        results = []
        if not features:
            return np.array([])

        # Check whether features are numpy arrays by testing the first element.
        first_elem = features[0]
        if isinstance(first_elem, np.ndarray):
            for agg_func in self.agg:
                if agg_func not in self.agg_funcs:
                    raise ValueError(f"Unknown aggregation function: {agg_func}")
                aggregated = self.agg_funcs[agg_func](features, axis=0)
                results.append(aggregated)
            return np.concatenate(results)
        else:
            for agg_func in self.agg:
                if agg_func not in self.agg_funcs:
                    raise ValueError(f"Unknown aggregation function: {agg_func}")
                results.append(self.agg_funcs[agg_func](features))
            return np.array(results)

    def get_feature_names(self) -> List[str]:
        raise NotImplementedError(
            "Get feature name method must be implemented by subclasses"
        )

    def citations(self) -> List[str]:
        return []

    def implementors(self) -> List[str]:
        return []


class PolymerPartFeaturizer:
    def __init__(self, calculator: Optional[BaseFeatureCalculator] = None):
        self.calculator = calculator

    def featurize(self, polymer) -> np.ndarray:
        raise NotImplementedError("Featurize method must be implemented by subclasses")

    def feature_labels(self) -> List[str]:
        if self.calculator:
            return [
                f"{label}_{self.__class__.__name__.lower()}"
                for label in self.calculator.feature_labels()
            ]
        else:
            return [self.__class__.__name__.lower()]