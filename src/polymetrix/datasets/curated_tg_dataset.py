import pandas as pd
from collections.abc import Collection
from typing import Optional, List
from polymetrix.constants import POLYMETRIX_PYSTOW_MODULE
from polymetrix.datasets import AbstractDataset


class CuratedGlassTempDataset(AbstractDataset):
    """Dataset for polymer glass transition temperature (Tg) data."""

    def __init__(
        self,
        version: str,
        url: str,
        feature_levels: List[str] = [
            "sidechainlevel",
            "backbonelevel",
            "fullpolymerlevel",
        ],
        subset: Optional[Collection[int]] = None,
    ):
        """Initialize the Tg dataset."""
        super().__init__()
        self._version = version
        self._url = url
        self._feature_levels = feature_levels
        self._all_feature_levels = [
            "sidechainlevel",
            "backbonelevel",
            "fullpolymerlevel",
        ]

        # Validate feature levels
        valid_levels = set(self._all_feature_levels)
        if not all(level in valid_levels for level in self._feature_levels):
            raise ValueError(
                f"feature_levels must be a subset of {valid_levels}, got {self._feature_levels}"
            )

        self._load_data(subset)

    def _load_data(self, subset: Optional[Collection[int]] = None):
        """Load and prepare the dataset."""
        csv_path = POLYMETRIX_PYSTOW_MODULE.ensure(
            "CuratedGlassTempDataset",
            self._version,
            url=self._url,
            name=".csv",
        )
        self._df = pd.read_csv(str(csv_path)).reset_index(drop=True)

        if subset is not None:
            self._df = self._df.iloc[subset].reset_index(drop=True)

        self._psmiles = self._df["PSMILES"].to_numpy()
        self._feature_names = [
            col
            for col in self._df.columns
            if any(
                col.startswith(f"{level}.features.") for level in self._feature_levels
            )
        ]
        self._label_names = [
            col for col in self._df.columns if col.startswith("labels.")
        ]
        self._meta_names = [col for col in self._df.columns if col.startswith("meta.")]

        self._features = self._df[self._feature_names].to_numpy()
        self._labels = self._df[self._label_names].to_numpy()
        self._meta_data = self._df[self._meta_names].to_numpy()

    @property
    def df(self) -> pd.DataFrame:
        """Return the underlying DataFrame."""
        return self._df

    @property
    def active_feature_levels(self) -> List[str]:
        return self._feature_levels

    def get_subset(self, indices: Collection[int]) -> "CuratedGlassTempDataset":
        return CuratedGlassTempDataset(
            version=self._version,
            url=self._url,
            feature_levels=self._feature_levels,
            subset=indices,
        )
