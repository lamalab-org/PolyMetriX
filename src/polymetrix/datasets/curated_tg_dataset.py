import pandas as pd
from collections.abc import Collection
from typing import Optional
from polymetrix.constants import POLYMETRIX_PYSTOW_MODULE
from polymetrix.datasets import AbstractDataset


class CuratedGlassTempDataset(AbstractDataset):
    """Dataset for polymer glass transition temperature (Tg) data."""

    def __init__(
        self,
        version: str,
        url: str,
        subset: Optional[Collection[int]] = None,
    ):
        """Initialize the Tg dataset.

        Args:
            version (str): Version of the dataset.
            url (str): URL to the dataset.
            subset (Optional[Collection[int]]): Indices to include in the dataset.
                If None, includes all entries.
        """
        super().__init__()
        self._version = version
        self._url = url

        # Get CSV path and load data properly
        csv_path = POLYMETRIX_PYSTOW_MODULE.ensure(
            "CuratedGlassTempDataset", 
            self._version, 
            url=self._url,
            name="data.csv",
        )
        self._df = pd.read_csv(str(csv_path)).reset_index(drop=True) 

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

    def get_subset(self, indices: Collection[int]) -> "CuratedGlassTempDataset":
        """Get a subset of the dataset.

        Args:
            indices (Collection[int]): Indices to include in the subset.

        Returns:
            CuratedGlassTempDataset: A new dataset containing only the specified indices.
        """
        return CuratedGlassTempDataset(
            version=self._version, 
            url=self._url, 
            subset=indices
        )