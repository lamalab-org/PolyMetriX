# PolyMetriX
PolyMetriX is a Python package designed for getting the polymer related featurizers that can be used for downstream tasks. The functionality of this package is that deals with the rdkit mol objects to get desired featurizers on full, sidechain and backbone polymer level. In addition, you can also use curated dataset for glass transition temperature (Tg) data for the polymers using this package.

# Citing PolyMetriX
We are currently in the process of writing a paper on PolyMetrix - we will update the citation information here once the paper is published.


# Features available in PolyMetriX 
PolyMetriX provides the following features:
- **NumHBondDonors & NumHBondAcceptors**: These featurizers count hydrogen bond donors and acceptors, respectively, and can provide insights into the molecule’s ability to form hydrogen bonds with water
- **NumRotatableBonds**: This featurizer counts the number of rotatable bonds in the polymer, which can provide insights into its flexibility and conformational freedom.
- **NumRings**: This featurizer counts the number of rings in the polymer, The number of rings in a polymer can affect its structural stability and interactions with other molecules.
- **NumNonAromaticRings**: This feature counts the number of non-aromatic rings in the polymer.
- **NumAromaticRings**: Aromatic rings can play a role in the molecule's stability and reactivity.
- **NumAtoms**: This feature counts the number of atoms in the polymer.
- **TopologicalSurfaceArea**: The topological polar surface area reflects the molecule’s polarity, affecting its interaction with polar solvents like water.
- **FractionBicyclicRings**: Fraction of bicyclic rings, The fraction represents the relative amount of these bicyclic units compared to other structural elements in the polymer chain. This can affect rigidity, thermal stability.
- **NumAliphaticHeterocycles**: The number of aliphatic heterocycles in a polymer refers to the quantity of non-aromatic cyclic structures containing at least one heteroatom (such as nitrogen, oxygen, or sulfur) within the polymer chain.
- **SlogPVSA1**: This featurizer represent the surface area of a molecule contributing to its solubility in octanol, which can be correlated with lipophilicity.
- **BalabanJIndex**: This featurizer calculates the Balaban connectivity index,which measures the molecular complexity and how connected its atoms are. Higher values indicate more complex and interconnected structures.
- **MolecularWeightFeaturizer**: Molecular weight can influence solubility, as smaller molecules generally dissolve more easily.
- **Sp3CarbonCountFeaturizer**: sp3 hybridized carbons can provide information about the molecule three-dimensional structure and its potential solubility.
- **Sp2CarbonCountFeaturizer**: sp2 hybridized carbons can provide information about the molecule aromaticity and potential reactivity.
- **MaxEStateIndex**: This featurizer maximum index of atoms in a molecule. It can provide insights into the electronic state and distribution of charge within the molecule.
- **SMR_VSA5**: This featurizer represents the sum of Crippen-Wildman molar refractivity of atoms with van der Waals surface area 2.45 - 2.75.
- **FpDensityMorgan1**: This featurizer represents the density information related to presence of substructures information in the morgran fingerprint.
- **HalogenCounts**: This featurizer counts the presence of halogen atoms like fluorine, chlorine, bromine, and iodine in the molecule.
- **BondCounts**: This featurizer counts the number of bonds in the polymer, which can provide insights into its structural complexity and reactivity.
- **BridgingRingsCount**: This featurizer counts the number of bridging rings, which can provide insights into the molecule’s structural stability and rigidity.
- **MaxRingSize**: This featurizer calculates the maximum ring size in the molecule.
- **HeteroatomCount**: This featurizer counts the number of heteroatoms except carbon and hydrogen, which are incorporated as rings in heterocyclic carbon compounds.
- **HeteroatomDensity**: This featurizer calculates the density of heteroatoms in the molecule.
- **HeteroatomDistanceStats**: This featurizer calculates the distance statistics of heteroatoms in the molecule.
- **NumSideChainFeaturizer**: This featurizer provides the number of sidechains in the polymer. The number of sidechains can significantly influence the polymer's properties, including its crystallinity, density.
- **NumBackBoneFeaturizer**: This featurizer provides the number of backbone atoms in the polymer.



# Classes available to get the featurizers
- **SideChainFeaturizer**: This class provides the featurizers for the sidechain of the polymer.
- **BackBoneFeaturizer**: This class provides the featurizers for the backbone of the polymer.
- **FullPolymerFeaturizer**: This class provides the featurizers for the full polymer.
- **MultipleFeaturizer**: This class provides multiple featurizers at once.



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
from polymetrix.data_loader import load_tg_dataset

data = load_tg_dataset('PolymerTg.csv')
```
The data will be a pandas dataframe of the Tg dataset.
