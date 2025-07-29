from typing import List
import numpy as np
from polymetrix.featurizers.base_featurizer import PolymerPartFeaturizer


class MultipleFeaturizer:
    def __init__(self, featurizers: List[PolymerPartFeaturizer]):
        self.featurizers = featurizers
        self._last_polymer = None

    def featurize(self, polymer) -> np.ndarray:
        self._last_polymer = polymer  # Store for label generation
        features = []
        for featurizer in self.featurizers:
            feature = featurizer.featurize(polymer)
            if isinstance(feature, (int, float)):
                feature = np.array([feature])
            features.append(feature.flatten())
        return np.concatenate(features)

    def feature_labels(self) -> List[str]:
        """Return feature labels with '_with_terminalgroups' suffix when applicable."""
        labels = []
        for featurizer in self.featurizers:
            featurizer_labels = featurizer.feature_labels()
            labels.extend(featurizer_labels)

        # Add terminal groups suffix if last polymer had terminal groups
        if (
            hasattr(self, "_last_polymer")
            and self._last_polymer
            and hasattr(self._last_polymer, "terminal_groups")
            and self._last_polymer.terminal_groups
        ):
            labels = [
                label.replace(
                    "_backbonefeaturizer", "_with_terminalgroups_backbonefeaturizer"
                )
                for label in labels
            ]

        return labels

    def citations(self) -> List[str]:
        citations = []
        for featurizer in self.featurizers:
            if hasattr(featurizer, "calculator") and featurizer.calculator:
                citations.extend(featurizer.calculator.citations())
        return list(set(citations))

    def implementors(self) -> List[str]:
        implementors = []
        for featurizer in self.featurizers:
            if hasattr(featurizer, "calculator") and featurizer.calculator:
                implementors.extend(featurizer.calculator.implementors())
        return list(set(implementors))
