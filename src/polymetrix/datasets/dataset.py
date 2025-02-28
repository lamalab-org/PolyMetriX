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
    def _load_data(self, subset: Optional[Collection[int]] = None):
        """Load and prepare the dataset-specific data.

        Args:
            subset (Optional[Collection[int]]): Indices to include in the dataset.
        """
        pass

    def get_subset(self, indices: Collection[int]) -> "AbstractDataset":
        """Get a subset of the dataset."""
        if not all(0 <= i < len(self) for i in indices):
            raise IndexError("Indices out of bounds.")
        subset = self.__class__()
        subset._features = self._features[indices]
        subset._labels = self._labels[indices]
        subset._meta_data = self._meta_data[indices]
        subset._psmiles = self._psmiles[indices] if self._psmiles is not None else None
        subset._feature_names = self._feature_names.copy()
        subset._label_names = self._label_names.copy()
        subset._meta_names = self._meta_names.copy()
        return subset

    @property
    def available_features(self) -> list[str]:
        return self._feature_names

    @property
    def available_labels(self) -> list[str]:
        return self._label_names

    @property
    def meta_info(self) -> list[str]:
        return self._meta_names

    @property
    def psmiles(self) -> np.ndarray:
        return self._psmiles

    def __len__(self):
        return len(self._features) if self._features is not None else 0

    def __iter__(self):
        return iter(self._features)

    def get_features(
        self, idx: Collection[int], feature_names: Optional[Collection[str]] = None
    ) -> np.ndarray:
        if feature_names is None:
            return self._features[np.array(idx)]
        col_indices = [self._feature_names.index(name) for name in feature_names]
        return self._features[np.array(idx)][:, col_indices]

    def get_labels(
        self, idx: Collection[int], label_names: Optional[Collection[str]] = None
    ) -> np.ndarray:
        if label_names is None:
            return self._labels[np.array(idx)]
        col_indices = [self._label_names.index(name) for name in label_names]
        return self._labels[np.array(idx)][:, col_indices]

    def get_meta(
        self, idx: Collection[int], meta_keys: Optional[Collection[str]] = None
    ) -> np.ndarray:
        if meta_keys is None:
            return self._meta_data[np.array(idx)]
        col_indices = [self._meta_names.index(name) for name in meta_keys]
        return self._meta_data[np.array(idx)][:, col_indices]
