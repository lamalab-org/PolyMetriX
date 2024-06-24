# PolyMetriX

PolyMetriX is a Python-based tool designed to calculate various molecular descriptors for polymers generated from PSMILES strings.
## Descriptors Calculated

The following descriptors are calculated:

- **nconf20_2**: Number of conformers within 20 kcal/mol of the lowest energy conformer
- **rotatable_bonds**: Number of rotatable bonds in the molecule
- **num_rings**: Total number of rings in the molecule
- **num_aromatic_rings**: Number of aromatic rings in the molecule
- **num_non_aromatic_rings**: Number of non-aromatic rings in the molecule
- **hbond_acceptors**: Number of hydrogen bond acceptors
- **hbond_donors**: Number of hydrogen bond donors
- **n_sc**: Number of side chains
- **len_sc**: Lengths of the side chains
- **n_bb**: Number of backbones
- **len_bb**: Lengths of the backbones


## Local Installation

We recommend that you create a virtual conda environment on your computer in which you install the dependencies for this package. To do so head over to [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and follow the installation instructions there.


<!-- ### Install latest release

```bash
pip install mattext
``` -->

### Install development version

Clone this repository (you need `git` for this, if you get a `missing command` error for `git` you can install it with `sudo apt-get install git`)

```bash
git clone https://github.com/lamalab-org/Poly_descriptors.git
cd polymetrix
```

```bash
pip install -e .
```
