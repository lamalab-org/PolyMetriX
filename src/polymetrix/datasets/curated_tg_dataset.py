import pandas as pd
from collections.abc import Collection
from typing import Optional, List
from polymetrix.constants import POLYMETRIX_PYSTOW_MODULE
from polymetrix.datasets import AbstractDataset


class CuratedGlassTempDataset(AbstractDataset):
    """Dataset for polymer glass transition temperature (Tg) data."""

    ALL_FEATURE_LEVELS = [
        "sidechainlevel",
        "backbonelevel",
        "fullpolymerlevel",
    ]
    FEATURE_PREFIX = "features."
    LABEL_PREFIX = "labels."
    META_PREFIX = "meta."

    DEFAULT_VERSION = "v1"
    DEFAULT_URL = "https://zenodo.org/records/14980914/files/LAMALAB_CURATED_Tg_structured.csv?download=1"

    def __init__(
        self,
        feature_levels: List[str] = ALL_FEATURE_LEVELS,
        subset: Optional[Collection[int]] = None,
    ):
        """Initialize the Tg dataset.
         Args:
            feature_levels (List[str]): Feature levels to include
            subset (Optional[Collection[int]]): Indices to include in the dataset
         """
        super().__init__()
        self._version = self.DEFAULT_VERSION
        self._url = self.DEFAULT_URL
        self._feature_levels = feature_levels

        # Validate feature levels using set operations
        if not set(self._feature_levels).issubset(self.ALL_FEATURE_LEVELS):
            raise ValueError(
                f"feature_levels must be a subset of {self.ALL_FEATURE_LEVELS}, "
                f"got {self._feature_levels}"
            )

        self._load_data(subset)

    def _load_data(self, subset: Optional[Collection[int]] = None):
        """Load and prepare the dataset."""
        csv_path = POLYMETRIX_PYSTOW_MODULE.ensure(
            "CuratedGlassTempDataset",
            self._version,
            url=self._url,
        )
        self._df = pd.read_csv(str(csv_path)).reset_index(drop=True)

        if subset is not None:
            self._df = self._df.iloc[subset].reset_index(drop=True)

        self._psmiles = self._df["PSMILES"].to_numpy()

        allowed_prefixes = [
            f"{level}.{self.FEATURE_PREFIX}" for level in self._feature_levels
        ]
        self._feature_names = self._filter_columns(allowed_prefixes)

        self._label_names = self._filter_columns([self.LABEL_PREFIX])
        self._meta_names = self._filter_columns([self.META_PREFIX])

        self._features = self._df[self._feature_names].to_numpy()
        self._labels = self._df[self._label_names].to_numpy()
        self._meta_data = self._df[self._meta_names].to_numpy()

    def _filter_columns(self, prefixes: List[str]) -> List[str]:
        """Helper to filter columns by prefix(es)."""
        return [
            col
            for col in self._df.columns
            if any(col.startswith(prefix) for prefix in prefixes)
        ]

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def active_feature_levels(self) -> List[str]:
        return self._feature_levels

    def get_subset(self, indices: Collection[int]) -> "CuratedGlassTempDataset":
        return CuratedGlassTempDataset(
            feature_levels=self._feature_levels,
            subset=indices,
        )
