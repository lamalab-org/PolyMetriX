from typing import List
import numpy as np
from polymetrix.featurizers.base_featurizer import PolymerPartFeaturizer


class MultipleFeaturizer:
    def __init__(self, featurizers: List[PolymerPartFeaturizer]):
        self.featurizers = featurizers

    def featurize(self, polymer) -> np.ndarray:
        features = []
        for featurizer in self.featurizers:
            feature = featurizer.featurize(polymer)
            if isinstance(feature, (int, float)):
                feature = np.array([feature])
            features.append(feature.flatten())
        return np.concatenate(features)

    def feature_labels(self) -> List[str]:
        labels = []
        for featurizer in self.featurizers:
            labels.extend(featurizer.feature_labels())
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