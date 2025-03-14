<h1 align="center">
  PolyMetriX
</h1>
<p align="center">
    <a href="https://pypi.org/project/polymetrix">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/polymetrix" />
    </a>
    <a href="./LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/polymetrix" />
    </a>
    <a href='https://lamalab-org.github.io/PolyMetriX/'>
        <img src="https://img.shields.io/badge/docs-passing-brightgreen" alt="Documentation">
    </a>
    <a href="https://www.contributor-covenant.org">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" />
    </a>
</p>

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./docs/figures/overview-dark.png">
  <img alt="PolyMetriX Overview" src="./docs/figures/overview.png">
</picture>
</p>

*PolyMetriX* is a comprehensive Python library that powers the entire machine learning workflow for polymer informatics. From data preparation to feature engineering, it provides a unified framework for developing structure-property relationships in polymer science.

For more detailed information, see the [documentation](https://lamalab-org.github.io/PolyMetriX/).

## ✨ Installing PolyMetriX

**Prerequisites**

- **Python 3.10 or newer:**

```bash
pip install polymetrix
```
For more detailed installation instructions, see the [documentation](https://lamalab-org.github.io/PolyMetriX/installation/).

## Loading Curated Glass Transition Temperature Dataset

```python
# Import necessary modules
from polymetrix.datasets import CuratedGlassTempDataset

# Load the dataset
dataset = CuratedGlassTempDataset()
```

## Getting Features for Polymers

```python
from polymetrix.featurizers.polymer import Polymer
from polymetrix.featurizers.chemical_featurizer import MolecularWeight
from polymetrix.featurizers.sidechain_backbone_featurizer import FullPolymerFeaturizer

# initialize the FullPolymerFeaturizer class with required featurizers
featurizer = FullPolymerFeaturizer(MolecularWeight())

polymer = Polymer.from_psmiles('*CCCCCCNC(=O)c1ccc(C(=O)N*)c(Sc2ccccc2)c1')
result = featurizer.featurize(polymer)
```
For more detailed usage instructions, see the [documentation](https://lamalab-org.github.io/PolyMetriX/how_to_guides/).


# How to contribute

We welcome contributions to PolyMetriX! Please refer to the [contribution guidelines](https://lamalab-org.github.io/PolyMetriX/contributing/) for more information.
