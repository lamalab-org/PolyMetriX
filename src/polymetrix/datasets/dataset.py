from collections.abc import Collection
from typing import Optional
import numpy as np
from abc import ABC, abstractmethod


class AbstractDataset(ABC):
    """Base class for polymer datasets."""

    def __init__(self):
        """Initialize a dataset."""
        self._meta_data = None
        self._features = None
        self._labels = None
        self._psmiles = None
        self._feature_names = []
        self._label_names = []
        self._meta_names = []

    @abstractmethod
    def get_subset(self, indices: Collection[int]) -> "AbstractDataset":
        """Get a subset of the dataset."""
        pass

    @property
    def available_features(self) -> list[str]:
        """List of available feature names in the dataset."""
        return self._feature_names

    @property
    def available_labels(self) -> list[str]:
        """List of available label names in the dataset."""
        return self._label_names

    @property
    def meta_info(self) -> list[str]:
        """List of available metadata fields in the dataset."""
        return self._meta_names

    def __len__(self):
        """Return the number of entries in the dataset."""
        return len(self._features)

    def __iter__(self):
        """Iterate over the features in the dataset."""
        return iter(self._features)

    def get_features(
        self, idx: Collection[int], feature_names: Optional[Collection[str]] = None
    ) -> np.ndarray:
        """Get features for specified indices."""
        if feature_names is None:
            return self._features[idx]
        col_indices = [self._feature_names.index(name) for name in feature_names]
        return self._features[idx][:, col_indices]

    def get_labels(
        self, idx: Collection[int], label_names: Optional[Collection[str]] = None
    ) -> np.ndarray:
        """Get labels for specified indices."""
        if label_names is None:
            return self._labels[idx]
        col_indices = [self._label_names.index(name) for name in label_names]
        return self._labels[idx][:, col_indices]

    def get_meta(
        self, idx: Collection[int], meta_keys: Optional[Collection[str]] = None
    ) -> np.ndarray:
        """Get metadata for specified indices."""
        if meta_keys is None:
            return self._meta_data[idx]
        col_indices = [self._meta_names.index(name) for name in meta_keys]
        return self._meta_data[idx][:, col_indices]
