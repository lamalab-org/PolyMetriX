# Splitters
Splitters are used to split the dataset into training, validation, and test sets. The splitters are designed to ensure that the training, validation, and test sets are distinct and do not overlap. The splitter are used to prevent the data leakage and these splitters tests the model's generalization ability. In the PolyMetriX, we have implemented only **Tgsplitter**. The other splitters like **KennardStoneSplitter** and **LOCOCVSplitter** are imported from the MOFDScribe package. See documentation for MOFDScribe for more information on these splitters [here](https://mofdscribe.readthedocs.io/en/latest/api/featurizers.html).

## TgSplitter

Splitter that uses the glass transition temperature (Tg) to split the dataset.

This splitter sorts structures by their Tg values and groups them using quantile binning. The grouping helps ensure different folds contain distinct Tg ranges, creating stringent validation conditions. The splitter also ensures that different folds contain different ranges of Tg values. This splitter can also be called as property based extrapolation splitter.

### Example

```python
from polymetrix.datasets import CuratedGlassTempDataset
from polymetrix.splitters.splitters import TgSplitter

dataset = CuratedGlassTempDataset(version, url)

splitter = TgSplitter(
    ds=dataset,
    tg_q=np.linspace(0, 1, 3), # Quantile bins for Tg values
    shuffle=True,
    random_state=42
)

train_idx, valid_idx, test_idx = splitter.train_valid_test_split(frac_train=0.6, frac_valid=0.1)
print(f"Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")

```

The splitter outputs the indices for each subset, ensuring that the training, validation, and test sets contain distinct Tg ranges. The splitter also ensures that the training, validation, and test sets contain different ranges of Tg values.

## KennardStoneSplitter

Splitter that uses the Kennard-Stone algorithm to split the dataset. This method selects the most dissimilar data points as the training set and the remaining data points as the test set. The splitter ensures that the training set contains the most diverse data points. To know more about the Kennard-Stone algorithm, see the documentation [here](https://mofdscribe.readthedocs.io/en/latest/api/splitters.html).


### Example

```python
from polymetrix.datasets import CuratedGlassTempDataset
from mofdscribe.splitters.splitters import KennardStoneSplitter

dataset = CuratedGlassTempDataset(version, url)
feature_names = dataset.available_features 

splitter = KennardStoneSplitter(
    ds=dataset,
    feature_names=feature_names,  
    scale=True,  
    centrality_measure="mean",
    metric="euclidean"
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

Leave-One-Cluster-Out Cross-Validation (LOCOCV) splitter that uses the cluster information to split the dataset. This method ensures that the training and validation sets contain distinct clusters. There is no overlap between the training and validation sets, and the training set contains all clusters except the one in the validation set. To know more about the LOCOCV algorithm, see the documentation [here](https://mofdscribe.readthedocs.io/en/latest/api/splitters.html).

### Example

```python
from polymetrix.datasets import CuratedGlassTempDataset
from mofdscribe.splitters.splitters import LOCOCV

dataset = CuratedGlassTempDataset(version, url)
feature_names = dataset.available_features 

loco = LOCOCV(
    ds=dataset,
    feature_names=feature_names, 
    n_pca_components=2,  
    random_state=42,
    scaled=True  
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