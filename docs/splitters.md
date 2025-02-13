# Splitters

Classes that help performing the splitting of the dataset into training and validation sets.

## TgSplitter

Splitter that uses the glass transition temperature (Tg) to split the dataset.

This splitter sorts structures by their Tg values and groups them using quantile binning. The grouping helps ensure different folds contain distinct Tg ranges, creating stringent validation conditions. The splitter also ensures that different folds contain different ranges of Tg values.

### Example

```python
from polymetrix.datasets import CuratedGlassTempDataset
from polymetrix.splitters.splitters import TgSplitter

dataset = CuratedGlassTempDataset(version, url)

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

### Example

```python
from polymetrix.datasets import CuratedGlassTempDataset
from mofdscribe.splitters.splitters import KennardStoneSplitter

dataset = CuratedGlassTempDataset(version, url)

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

The splitter outputs the indices for each subset, using Kennard-Stone's deterministic selection to ensure the training set contains maximally diverse, space-filling samples that optimally represent the feature space. The validation/test sets then automatically inherit structurally distinct data points not selected in this representative training subset.

## LOCOCVSplitter

Leave-One-Cluster-Out Cross-Validation (LOCOCV) splitter that uses the cluster information to split the dataset. This method ensures that the training and validation sets contain distinct clusters.

### Example

```python
from polymetrix.datasets import CuratedGlassTempDataset
from mofdscribe.splitters.splitters import LOCOCV

dataset = CuratedGlassTempDataset(version, url)

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

