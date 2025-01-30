# Splitters

Classes that help performing the splitting of the dataset into training and validation sets.

## TgSplitter

Splitter that uses the glass transition temperature (Tg) to split the dataset.

This splitter sorts structures by their Tg values and groups them using quantile binning. The grouping helps ensure different folds contain distinct Tg ranges, creating stringent validation conditions. The splitter also ensures that different folds contains different ranges of Tg values.

### Initialization

```python
def init(
ds: AbstractDataset,
group_col: str = "Tg",
tg_q: Optional[Collection[float]] = None,
shuffle: bool = True,
random_state: Optional[Union[int, np.random.RandomState]] = None,
sample_frac: Optional[float] = 1.0,
stratification_col: Optional[Union[str, np.typing.ArrayLike]] = None,
center: callable = np.median,
q: Collection[float] = (0, 0.25, 0.5, 0.75, 1),
sort_by_len: bool = True,
)
```

**Parameters**:

- `ds` (AbstractDataset): Input dataset containing Tg values
- `group_col` (str): Column name storing Tg values (default: "Tg")
- `tg_q` (Collection[float], optional): Quantiles for Tg binning. Default behavior:
  - 2 bins for train/test splits
  - 3 bins for train/valid/test splits
  - k bins for k-fold
- `shuffle` (bool): Enable index shuffling (default: True)
- `random_state`: Seed/state for reproducible shuffling
- `sample_frac`: Fraction of dataset to use (1.0 = full dataset)
- `stratification_col`: Column for stratified splitting (categorical or quantile-binned)
- `center`: Central tendency measure for continuous stratification (default: median)
- `q`: Quantiles for general quantile binning (default: [0, 0.25, 0.5, 0.75, 1])
- `sort_by_len`: Sort splits by size (default: True)

### Implementation Details

```python
from polymetrix.datasets import GlassTempDataset
from polymetrix.data_loader import load_tg_dataset
from polymetrix.splitters import TgSplitter

df = load_tg_dataset('PolymerTg.csv')
dataset = GlassTempDataset(df=df)

splitter = TgSplitter(
    ds=dataset,
    group_col="labels.Exp_Tg(K)",
    shuffle=True,
    random_state=42,
    sample_frac=1.0,
)

train_idx, valid_idx, test_idx = splitter.train_valid_test_split(frac_train=0.6, frac_valid=0.1)
print(f"Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")

```

The resulting output will be the number of indices in the training, validation, and test sets. The splitter will ensure that the Tg values in the training, validation, and test sets are distinct and do not overlap.

## KennardStoneSplitter

Splitter that uses the Kennard-Stone algorithm to split the dataset. This method selects the most dissimilar data points as the training set and the remaining data points as the test set. The splitter ensures that the training set contains the most diverse data points.

### Initialization

```python
def init(
ds: AbstractDataset,
feature_names: list[str],
shuffle: bool = True,
random_state: Optional[Union[int, np.random.RandomState]] = None,
sample_frac: Optional[float] = 1.0,
scale: bool = True,
centrality_measure: str = "mean",
metric: Union[Callable, str] = "euclidean",
ascending: bool = False,
) -> None
```

**Parameters**:

- `ds` (AbstractDataset): Input dataset containing structural features
- `feature_names` (list[str]): Feature columns used for distance calculations
- `shuffle` (bool): Enable index shuffling within splits (default: True)
- `random_state`: Seed/state for reproducible shuffling
- `sample_frac`: Fraction of dataset to use (1.0 = full dataset)
- `scale` (bool): Apply z-score normalization to features (default: True)
- `centrality_measure` (str): Initial sample selection strategy:
  - "mean": Maximize distance from feature mean
  - "median": Maximize distance from feature median
  - "random": Random initial point selection
    (default: "mean")
- `metric`: Distance metric for feature comparison (default: "euclidean")
- `ascending` (bool): Reverse sample order (default: False)

### Implementation Example

```python
from polymetrix.datasets import GlassTempDataset
from polymetrix.splitters import KennardStoneSplitter
from polymetrix.data_loader import load_tg_dataset

df = load_tg_dataset('PolymerTg.csv')
dataset = GlassTempDataset(df=df)

splitter = KennardStoneSplitter(
    ds=dataset,
    feature_names=dataset.available_features,
    shuffle=True,
    random_state=42,
    sample_frac=1.0,
    scale=True,
    centrality_measure="mean",
    metric="euclidean",
    ascending=False,
)

# Different splitting methods

# Train-test split
train_idx, test_idx = splitter.train_test_split(frac_train=0.7) 

# Train-validation-test split
train_idx, valid_idx, test_idx = splitter.train_valid_test_split(frac_train=0.7, frac_valid=0.15)

# 5-fold cross-validation
for fold, (train_index, test_index) in enumerate(splitter.k_fold(k=5)):
    print(f"Fold {fold}: Train: {len(train_index)}, Test: {len(test_index)}")
```

The resulting output will be the number of indices in the training, validation, and test sets. The splitter will ensure that the training, validation, and test sets contain the most diverse data points.

## LOCOCVSplitter

Leave-One-Cluster-Out Cross-Validation (LOCOCV) splitter that uses the cluster information to split the dataset. This method ensures that the training and validation sets contain distinct clusters.

### Initialization

```python
def init(
ds: AbstractDataset,
feature_names: list[str],
shuffle: bool = True,
random_state: Optional[Union[int, np.random.RandomState]] = None,
sample_frac: Optional[float] = 1.0,
scaled: bool = True,
n_pca_components: Optional[int] = "mle",
pca_kwargs: Optional[dict[str, Any]] = None,
kmeans_kwargs: Optional[dict[str, Any]] = None,
):
```


**Parameters**:

- `ds` (AbstractDataset): Input dataset containing structural features
- `feature_names` (list[str]): Feature columns used for clustering
- `shuffle` (bool): Enable index shuffling within clusters (default: True)
- `random_state`: Seed/state for reproducible clustering/shuffling
- `sample_frac`: Fraction of dataset to use (1.0 = full dataset)
- `scaled` (bool): Apply z-score normalization before PCA (default: True)
- `n_pca_components`: PCA dimensions (int) or "mle" for automatic selection (default: "mle")
- `pca_kwargs`: Custom parameters for sklearn PCA implementation
- `kmeans_kwargs`: Custom parameters for sklearn KMeans clustering

### Implementation Example

```python
from polymetrix.datasets import GlassTempDataset
from polymetrix.splitters import LOCOCV
from polymetrix.data_loader import load_tg_dataset

df = load_tg_dataset('PolymerTg.csv')

dataset = GlassTempDataset(df=df)

loco = LOCOCV(
    ds=dataset,
    feature_names=["features.num_atoms_sidechainfeaturizer_sum", 'features.num_rotatable_bonds_fullpolymerfeaturizer', 'features.num_rings_fullpolymerfeaturizer']
    shuffle=True,
    random_state=42,  
    scaled=True,  
    n_pca_components="mle",  
)

# Different splitting methods

# Train-test split  
train_idx, test_idx = loco.train_test_split(frac_train=0.7)

# Train-validation-test split
train_idx, valid_idx, test_idx = loco.train_valid_test_split(frac_train=0.7, frac_valid=0.15)

# 5-fold cross-validation
for fold, (train_index, test_index) in enumerate(loco.k_fold(k=5)):
    print(f"Fold {fold}: Train: {len(train_index)}, Test: {len(test_index)}")
```
The resulting output will be the number of indices in the training, validation, and test sets. The splitter will ensure that the training and validation sets contain distinct clusters.

## ClusterSplitter
Cluster-based dataset splitter using PCA and k-means clustering to create structurally distinct groups. 

### Initialization

```python
def init(
ds: AbstractDataset,
feature_names: list[str],
shuffle: bool = True,
random_state: Optional[Union[int, np.random.RandomState]] = None,
sample_frac: Optional[float] = 1.0,
stratification_col: Optional[Union[str, np.typing.ArrayLike]] = None,
center: callable = np.median,
q: Collection[float] = (0, 0.25, 0.5, 0.75, 1),
sort_by_len: bool = False,
scaled: bool = True,
n_pca_components: Optional[Union[int, str]] = "mle",
n_clusters: int = 4,
pca_kwargs: Optional[dict[str, Any]] = None,
kmeans_kwargs: Optional[dict[str, Any]] = None,
):
```


**Parameters**:

- `ds` (AbstractDataset): Input dataset containing structural features
- `feature_names` (list[str]): Feature columns used for clustering
- `n_clusters` (int): Number of k-means clusters (default: 4)
- `scaled` (bool): Apply z-score normalization pre-PCA (default: True)
- `n_pca_components`: PCA dimensions (int) or "mle" for automatic selection
- `pca_kwargs`: Custom parameters for sklearn PCA implementation
- `kmeans_kwargs`: Custom parameters for sklearn KMeans clustering
- `shuffle` (bool): Enable index shuffling within clusters (default: True)
- `random_state`: Seed for reproducible clustering/shuffling
- `sample_frac`: Fraction of dataset to use (1.0 = full dataset)

### Implementation Example

```python
from polymetrix.datasets import GlassTempDataset
from polymetrix.splitters import ClusterSplitter
from polymetrix.data_loader import load_tg_dataset

df = load_tg_dataset('PolymerTg.csv')
dataset = GlassTempDataset(df=df)

splitter = ClusterSplitter(
    ds=dataset,
    feature_names=["features.num_atoms_sidechainfeaturizer_sum", 'features.num_rotatable_bonds_fullpolymerfeaturizer', 'features.num_rings_fullpolymerfeaturizer']
    shuffle=True,
    random_state=42,  
    scaled=True,  
    n_pca_components="mle",  
    n_clusters=4,
)

# Different splitting methods

# Train-test split
train_idx, test_idx = splitter.train_test_split(frac_train=0.7)

# Train-validation-test split
train_idx, valid_idx, test_idx = splitter.train_valid_test_split(frac_train=0.7, frac_valid=0.15)

# 5-fold cross-validation
for fold, (train_index, test_index) in enumerate(splitter.k_fold(k=5)):
    print(f"Fold {fold}: Train: {len(train_index)}, Test: {len(test_index)}")
```
The resulting output will be the number of indices in the training, validation, and test sets. The splitter will ensure that the training and validation sets contain distinct clusters.
