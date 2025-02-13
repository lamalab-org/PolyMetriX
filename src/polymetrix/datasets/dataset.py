from collections.abc import Collection
from typing import Optional
import numpy as np


class AbstractDataset:
    """Base class for polymer datasets."""

    def __init__(self):
        """Initialize a dataset."""
        self._meta_data = None
        self._features = None
        self._labels = None
        self._psmiles = None

    def get_subset(self, indices: Collection[int]) -> "AbstractDataset":
        """Get a subset of the dataset.

        Args:
            indices (Collection[int]): Indices to include in the subset.

        Returns:
            AbstractDataset: A new dataset containing only the specified indices.
        """
        raise NotImplementedError()

    @property
    def available_features(self) -> list[str]:
        """List of available feature names in the dataset.

        Returns:
            List[str]: List of feature names.
        """
        raise NotImplementedError()

    @property
    def available_labels(self) -> list[str]:
        """List of available label names in the dataset.

        Returns:
            List[str]: List of label names.
        """
        raise NotImplementedError()

    @property
    def meta_info(self) -> list[str]:
        """List of available metadata fields in the dataset.

        Returns:
            List[str]: List of metadata field names.
        """
        raise NotImplementedError()

    def __len__(self):
        """Return the number of entries in the dataset."""
        return len(self._features)

    def __iter__(self):
        """Iterate over the features in the dataset."""
        return iter(self._features)

    def get_features(
        self, idx: Collection[int], feature_names: Optional[Collection[str]] = None
    ) -> np.ndarray:
        """Get features for specified indices.

        Args:
            idx (Collection[int]): Indices of entries.
            feature_names (Optional[Collection[str]]): Names of features to return.
                If None, returns all available features.

        Returns:
            np.ndarray: Array of feature values.
        """
        raise NotImplementedError()

    def get_labels(
        self, idx: Collection[int], label_names: Optional[Collection[str]] = None
    ) -> np.ndarray:
        """Get labels for specified indices.

        Args:
            idx (Collection[int]): Indices of entries.
            label_names (Optional[Collection[str]]): Names of labels to return.
                If None, returns all available labels.

        Returns:
            np.ndarray: Array of label values.
        """
        raise NotImplementedError()

    def get_meta(
        self, idx: Collection[int], meta_keys: Optional[Collection[str]] = None
    ) -> np.ndarray:
        """Get metadata for specified indices.

        Args:
            idx (Collection[int]): Indices of entries.
            meta_keys (Optional[Collection[str]]): Names of metadata fields to return.
                If None, returns all available metadata.

        Returns:
            np.ndarray: Array of metadata values.
        """
        raise NotImplementedError()