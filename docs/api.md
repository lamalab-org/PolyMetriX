The installation and usage of the gettting the featurizers and dataset is described in the following sections.

# Installation
The most recent code can be installed directly from GitHub with:
```shell
$ pip install git+https://github.com/lamalab-org/PolyMetriX.git
```

To install in development mode, use the following:

```shell
$ git clone https://github.com/lamalab-org/PolyMetriX.git
$ cd polymetrix
$ pip install -e .
```

# Usage example for getting the single featurizer for the full polymer level
```python
from polymetrix.featurizer import FullPolymerFeaturizer, MolecularWeightFeaturizer, 

# initialize the FullPolymerFeaturizer class with required featurizers
featurizer = FullPolymerFeaturizer(MolecularWeightFeaturizer())

polymer = Polymer.from_psmiles('*CCCCCCNC(=O)c1ccc(C(=O)N*)c(Sc2ccccc2)c1')
result = featurizer.featurize(polymer)
```
The result will be a NumPy array of MolecularWeightFeaturizer value for the given polymer.


# Example for getting the multiple featurizers for the full polymer level
```python
from polymetrix.featurizer import FullPolymerFeaturizer, MultipleFeaturizer, MolecularWeightFeaturizer, NumHBondDonors, NumHBondAcceptors, NumRotatableBonds

# initialize the FullPolymerFeaturizer class with required featurizers
mol_weight_featurizer = FullPolymerFeaturizer(MolecularWeightFeaturizer())
hbond_donors = FullPolymerFeaturizer(NumHBondDonors())
hbond_acceptors = FullPolymerFeaturizer(NumHBondAcceptors())
rotatable_bonds = FullPolymerFeaturizer(NumRotatableBonds())

featurizer = MultipleFeaturizer([mol_weight_featurizer, hbond_donors, hbond_acceptors, rotatable_bonds])
polymer = Polymer.from_psmiles('*CCCCCCNC(=O)c1ccc(C(=O)N*)c(Sc2ccccc2)c1')
result = featurizer.featurize(polymer)
```
The result will be a NumPy array of mol_weight_featurizer, hbond_donors, hbond_acceptors, rotatable_bonds values for the given polymer.

# Example for getting the featurizers for the sidechain level of the polymer that is number and length of sidechains
```python
from polymetrix.featurizer import SideChainFeaturizer, NumSideChainFeaturizer, MultipleFeaturizer, NumAtoms

# initialize the SideChainFeaturizer class with required featurizers
num_sidechains = NumSideChainFeaturizer()
sidechain_length = SideChainFeaturizer(NumAtoms(agg=["sum", "mean", "max", "min"]))

featurizer = MultipleFeaturizer([num_sidechains, sidechain_length])
polymer = Polymer.from_psmiles('*CCCCCCCCOc1ccc(C(c2ccc(O*)cc2)(C(F)(F)F)C(F)(F)F)cc1')
result = featurizer.featurize(polymer)
```
The result will be a NumPy array of num_sidechains and sidechain_length values for the given polymer.


# Example for getting the featurizers for the backbone level of the polymer that is number and length of backbone atoms
```python
from polymetrix.featurizer import BackBoneFeaturizer, NumBackBoneFeaturizer, MultipleFeaturizer, NumAtoms

# initialize the BackBoneFeaturizer class with required featurizers
num_backbones = NumBackBoneFeaturizer()
backbone_length = BackBoneFeaturizer(NumAtoms(agg=["sum"])) # Polymer cannot have more than one backbone

featurizer = MultipleFeaturizer([num_backbones, backbone_length])
polymer = Polymer.from_psmiles('*CCCCCCCCOc1ccc(C(c2ccc(O*)cc2)(C(F)(F)F)C(F)(F)F)cc1')
result = featurizer.featurize(polymer)
```
The result will be a NumPy array of num_backbones and backbone_length values for the given polymer.


# Example for getting featurizers for the sidechain and backbone level of the polymer
```python
from polymetrix.featurizer import SideChainFeaturizer, BackBoneFeaturizer, NumHBondDonors, NumHBondAcceptors, Sp3CarbonCountFeaturizer, MultipleFeaturizer

# initialize the SideChainFeaturizer and BackBoneFeaturizer class with required featurizers
sidechain_num_hbond_donors = SideChainFeaturizer(NumHBondDonors())
sidechain_num_hbond_acceptors = SideChainFeaturizer(NumHBondAcceptors())
sidechain_sp3_carbon_count = SideChainFeaturizer(Sp3CarbonCountFeaturizer())
backbone_num_hbond_donors = BackBoneFeaturizer(NumHBondDonors())
backbone_num_hbond_acceptors = BackBoneFeaturizer(NumHBondAcceptors())
backbone_sp3_carbon_count = BackBoneFeaturizer(Sp3CarbonCountFeaturizer())

featurizer = MultipleFeaturizer([sidechain_num_hbond_donors, sidechain_num_hbond_acceptors, sidechain_sp3_carbon_count, backbone_num_hbond_donors, backbone_num_hbond_acceptors, backbone_sp3_carbon_count])
polymer = Polymer.from_psmiles('*CCCCCCCCOc1ccc(C(c2ccc(O*)cc2)(C(F)(F)F)C(F)(F)F)cc1')
result = featurizer.featurize(polymer)
```
The result will be a NumPy array of sidechain_num_hbond_donors, sidechain_num_hbond_acceptors, sidechain_sp3_carbon_count, backbone_num_hbond_donors, backbone_num_hbond_acceptors, backbone_sp3_carbon_count values for the given polymer.

# Additionally, you can load the curated dataset for glass transition temperature (Tg) data for the polymers using this package.
```python
# Import necessary modules
from polymetrix.datasets import CuratedGlassTempDataset

# Load the dataset
dataset = CuratedGlassTempDataset(version=version, url=url)
```
The dataset will be a class object that contains the data for the curated dataset for glass transition temperature (Tg) data for the polymers along with chemical and topoligical featurizers for the polymers.
