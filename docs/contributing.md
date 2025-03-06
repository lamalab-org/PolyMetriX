# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

Please note we have a code of conduct, please follow it in all your interactions with the project.

## How to Contribute
### 1. Fork the Repository
Fork the repository by clicking on the 'Fork' button in the top right corner of the repository page. This will create a copy of this repository in your GitHub account.

### 2. Clone the Repository
Clone the forked repository to your local machine. Open a terminal and run the following command:
```bash
git clone
```

### 3. Create a New Branch
Create a new branch to work on the changes. Run the following command to create a new branch:
```bash
git checkout -b <branch-name>
```

### 4. Make Changes
Make the necessary changes in the codebase.

### 5. Commit Changes
Commit the changes to the branch. Run the following command to commit the changes:
```bash
git commit -m "Your commit message"
```

### 6. Push Changes
Push the changes to the forked repository. Run the following command to push the changes:
```bash
git push origin <branch-name>
```

### 7. Create a Pull Request
Go to the forked repository on GitHub and click on the 'New Pull Request' button. Fill in the details of the pull request and submit it.


# What You Can Contribute
You can contribute to the project in the following ways:
- *Featurizers*: Enhance the polymer analysis capabilities by adding new featurizers.
- *Datasets*: Contribute new polymer datasets (e.g., glass transition temperature, tensile strength).
- *Documentation*: Improve the documentation to make it more user-friendly.
- *Feature Requests*: Suggest new features that you would like to see in the project.

# Detailed Contribution Guidelines
### Contributing Featurizers
You can enhance the polymer analysis capabilities by adding new featurizers to `PolyMetriX/src/polymetrix/featurizer.py`. Featurizers compute specific properties of polymers. Here’s how:

1. Create a new Featurizer class in `PolyMetriX/src/polymetrix/featurizer.py`:
    - Subclass `BaseFeatureCalculator` or `PolymerPartFeaturizer` (e.g., for sidechains, backbone, or full polymer).
    - Implement the `calculate()` method (or `featurize()` for polymer-specific featurizers) to compute your feature.
    - Define `feature_base_labels()` to name your feature(s).

Example:
```python   
class MyNewFeaturizer(BaseFeatureCalculator):
    def calculate(self, polymer: Polymer) -> np.ndarray:
        # Compute your feature here
        return feature_values

    def feature_base_labels(self) -> List[str]:
        return ['my_new_feature']
```

2. Test your featurizer by adding a test case in `PolyMetriX/tests/test_featurizer.py`. We use `pytest` to run the tests.

### Contributing Datasets
You can contribute new polymer datasets (e.g., glass transition temperature, tensile strength) to the `datasets` folder. Datasets should follow a standard format and use `pystow` for retrieval.
1. Create a Dataset Class:
    - Subclass AbstractDataset in PolyMetriX/src/polymetrix/datasets/.
    - Define the dataset’s properties (e.g., PSMILES, features, labels).
    - We use zenodo for versioning datasets.
    - Use `pystow` to load the dataset from a URL.

Example:
```python
class CuratedDensityDataset(AbstractDataset):
    def __init__(self, version: str, url: str, subset: Optional[Collection[int]] = None):
        super().__init__()
        self._version = version
        self._url = url
        csv_path = POLYMETRIX_PYSTOW_MODULE.ensure("CuratedDensityDataset", version, url=url, name=".csv")
        self._df = pd.read_csv(csv_path).reset_index(drop=True)
        if subset is not None:
            self._df = self._df.iloc[subset].reset_index(drop=True)
        self._psmiles = self._df["PSMILES"].to_numpy()
        self._features = self._df[[col for col in self._df.columns if col.startswith("features.")]].to_numpy()
        self._labels = self._df[[col for col in self._df.columns if col.startswith("labels.")]].to_numpy()
```
2. Dataset Format
    - CSV columns must include:
        - PSMILES: Polymer SMILES string.
        - `features.<name>`: Feature columns (e.g., features.molecular_weight).
        - `labels.<name>`: Label columns (e.g., labels.density).
        - Optional:`meta.<name>` for metadata columns.

3. Test your dataset by adding a test case in `PolyMetriX/tests/test_datasets.py`. We use `pytest` to run the tests.

4. Submit a pull request to the main repository.

### Contributing Documentation
You can contribute to the project’s documentation by improving the existing documentation or adding new sections. The documentation is located in the `docs` folder.

1. Edit the existing documentation or add new documentation in the `docs` folder.
2. Submit a pull request to the main repository.

The project uses `mkdocs` to generate the documentation. You can preview the documentation locally by running the following command:
```bash
mkdocs serve
```
That's it! You have successfully contributed to the project.

