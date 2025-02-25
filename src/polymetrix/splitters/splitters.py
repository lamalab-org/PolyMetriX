from mofdscribe.splitters.splitters import BaseSplitter
from mofdscribe.splitters.utils import quantile_binning
from typing import Collection, Optional, Union
import numpy as np
from polymetrix.datasets.dataset import AbstractDataset


class TgSplitter(BaseSplitter):
    """Splitter based on Tg values"""

    def __init__(
        self,
        ds: AbstractDataset,
        tg_q: Optional[Collection[float]] = None,
        label_name: str = "labels.Exp_Tg(K)",
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        **kwargs,
    ) -> None:
        """Initialize TgSplitter

        Args:
            ds: Dataset to split
            tg_q: Quantiles to bin Tg values into groups
            label_name: Name of the label to use for splitting
            shuffle: Whether to shuffle the dataset
            random_state: Random state for shuffling
            **kwargs: Additional arguments to pass to BaseSplitter
        """
        self._grouping_q = tg_q
        self._label_name = label_name
        super().__init__(ds=ds, shuffle=shuffle, random_state=random_state, **kwargs)

    def _get_groups(self) -> Collection[int]:
        """Bin Tg values into quantile-based groups"""
        tg_values = self._ds.get_labels(
            idx=range(len(self._ds)), label_names=[self._label_name]
        ).flatten()
        return quantile_binning(tg_values, self._grouping_q)
