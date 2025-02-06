import pandas as pd
import numpy as np
from collections.abc import Collection
from typing import Optional
from polymetrix.datasets import AbstractDataset


class GlassTempDataset(AbstractDataset):
    """Dataset for polymer glass transition temperature (Tg) data."""

    def __init__(
        self,
        df: pd.DataFrame,
        subset: Optional[Collection[int]] = None,
    ):
        """Initialize the Tg dataset.

        Args:
            df (pd.DataFrame): DataFrame containing the dataset.
            subset (Optional[Collection[int]]): Indices of entries to include.
                If None, includes all entries.
        """
        super().__init__()

        self._df = df.copy()

        if subset is not None:
            self._df = self._df.iloc[subset].reset_index(drop=True)

        self._psmiles = self._df["PSMILES"].to_numpy()

        self._feature_names = [
            col for col in self._df.columns if col.startswith("features.")
        ]
        self._label_names = [
            col for col in self._df.columns if col.startswith("labels.")
        ]
        self._meta_names = [col for col in self._df.columns if col.startswith("meta.")]

        self._features = self._df[self._feature_names].to_numpy()
        self._labels = self._df[self._label_names].to_numpy()
        self._meta_data = self._df[self._meta_names].to_numpy()

    @property
    def psmiles(self):
        return self._psmiles

    def get_subset(self, indices: Collection[int]) -> "GlassTempDataset":
        """Get a subset of the dataset.

        Args:
            indices (Collection[int]): Indices to include in the subset.

        Returns:
            GlassTempDataset: A new dataset containing only the specified indices.
        """
        return GlassTempDataset(df=self._df, subset=indices)

    @property
    def available_features(self) -> list[str]:
        """Get list of available feature names.

        Returns:
            List[str]: List of feature names.
        """
        return self._feature_names

    @property
    def available_labels(self) -> list[str]:
        """Get list of available label names.

        Returns:
            List[str]: List of label names.
        """
        return self._label_names

    @property
    def meta_info(self) -> list[str]:
        """Get list of available metadata fields.

        Returns:
            List[str]: List of metadata field names.
        """
        return self._meta_names

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
        features = feature_names if feature_names is not None else self._feature_names
        return self._df.iloc[idx][features].to_numpy()

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
        labels = label_names if label_names is not None else self._label_names
        return self._df.iloc[idx][labels].to_numpy()

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
        meta = meta_keys if meta_keys is not None else self._meta_names
        return self._df.iloc[idx][meta].to_numpy()
