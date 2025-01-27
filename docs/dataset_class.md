# Dataset Class
## Glass Temperature Dataset for Polymers

```python
from polymetrix.data_loader import load_tg_dataset
from polymetrix.datasets import GlassTempDataset

df = load_tg_dataset('PolymerTg.csv')

dataset = GlassTempDataset(df=df)

print("Available features:", dataset.available_features)
print("Available labels:", dataset.available_labels)
print("Available metadata:", dataset.meta_info)
```

This will output the list of available features, labels, and metadata for the dataset.

## Output
```shell
## List of Available Features
Available features: ['features.num_atoms_sidechainfeaturizer_sum', 'features.num_atoms_sidechainfeaturizer_mean', 'features.num_atoms_sidechainfeaturizer_max', 'features.num_atoms_sidechainfeaturizer_min', 'features.numsidechainfeaturizer', 'features.num_atoms_backbonefeaturizer', 'features.numbackbonefeaturizer', 'features.num_hbond_donors_fullpolymerfeaturizer', 'features.num_hbond_acceptors_fullpolymerfeaturizer', 'features.num_rotatable_bonds_fullpolymerfeaturizer', 'features.num_rings_fullpolymerfeaturizer', 'features.num_non_aromatic_rings_fullpolymerfeaturizer', 'features.num_aromatic_rings_fullpolymerfeaturizer', 'features.topological_surface_area_fullpolymerfeaturizer', 'features.fraction_bicyclic_rings_fullpolymerfeaturizer', 'features.num_aliphatic_heterocycles_fullpolymerfeaturizer', 'features.slogp_vsa1_fullpolymerfeaturizer', 'features.balaban_j_index_fullpolymerfeaturizer', 'features.molecular_weight_fullpolymerfeaturizer', 'features.sp3_carbon_count_fullpolymerfeaturizer', 'features.sp2_carbon_count_fullpolymerfeaturizer', 'features.max_estate_index_fullpolymerfeaturizer', 'features.smr_vsa5_fullpolymerfeaturizer', 'features.fp_density_morgan1_fullpolymerfeaturizer', 'features.total_halogens_fullpolymerfeaturizer', 'features.fluorine_count_fullpolymerfeaturizer', 'features.chlorine_count_fullpolymerfeaturizer', 'features.bromine_count_fullpolymerfeaturizer', 'features.single_bonds_fullpolymerfeaturizer', 'features.double_bonds_fullpolymerfeaturizer', 'features.triple_bonds_fullpolymerfeaturizer', 'features.bridging_rings_count_fullpolymerfeaturizer', 'features.max_ring_size_fullpolymerfeaturizer', 'features.heteroatom_density_fullpolymerfeaturizer', 'features.heteroatom_count_fullpolymerfeaturizer', 'features.heteroatom_distance_mean_fullpolymerfeaturizer', 'features.heteroatom_distance_min_fullpolymerfeaturizer', 'features.heteroatom_distance_max_fullpolymerfeaturizer', 'features.heteroatom_distance_sum_fullpolymerfeaturizer']
## List of Available Labels
Available labels: ['labels.Exp_Tg(K)']
## List of Available Metadata
Available metadata: ['meta.polymer', 'meta.PSMILES', 'meta.source', 'meta.tg_range', 'meta.number of points', 'meta.reliability', 'meta.stdev']
```

since, the dataset has been curated for the glass transition temperature (Tg) data for the polymers, the available labels are `labels.Exp_Tg(K)` and the available features are the list of features that are available in the dataset. In addition this dataset also contains metadata information about the polymer, PSMILES, source, tg_range, number of points, reliability, and standard deviation of the data.

The `meta.source`  contain the following names of the sources from which the data has been obtained for Tg dataset along with links to the sources:
- `Schrodinger` - [Schrodinger](https://pubs.acs.org/doi/10.1021/acsapm.0c00524)
- `Mattioni` - [Mattioni](https://pubs.acs.org/doi/10.1021/ci010062o)
- `Uchicago` - [Uchicago](https://pppdb.uchicago.edu/tg)
- `Liu` - [Liu](https://link.springer.com/article/10.1007/s00396-009-2035-y)
- `Nguyen` - [Nguyen](https://pubs.acs.org/doi/10.1021/acs.iecr.2c01302)
- `Wu` - [Wu](https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fpolb.24117&file=polb24117-sup-0001-suppinfo1.pdf)
- `Qiu` - [Qiu](https://github.com/HKQiu/PPP-1_PredictionTg4Polyimides/blob/main/Train%20data/GNN%E6%95%B0%E6%8D%AE%E5%BA%93.csv)
- `GREA` - [GREA]( https://github.com/liugangcode/GREA/blob/main/data/tg_prop/raw/tg_raw.csv)
- `Xie` - [Xie](https://github.com/figotj/Polymer_Tg_/blob/main/Data/32_Conjugate_Polymer.txt)

For the same polymer there exists different glass transition temperature values from different sources. For this reason, we have considered median value of the Tg values for the same polymer from different sources as the final Tg value for the polymer and provided the reliability of the data in the `meta.reliability` column. The reliability of the data is assigned on the occurence of polymer and Z-score ≤ 2. The reliability of the data is categorized into three categories:
- `Black` - This category means we don't know the reliability of the data because they are unique polymers.
- `Yellow` - This category means the data is moderately reliable beacuse the polymer has two different Tg values from different sources and Z-score ≤ 2.
- `Gold` - This category means the data is highly reliable because the polymer has more than two different Tg values from different sources and Z-score ≤ 2.

The `meta.tg_range` column contains the range of the glass transition temperature for the polymer for multiple sources. 
The `meta.number of points` column contains the number of data points for the polymer that has different Tg values from different sources.
The `meta.stdev` column contains the standard deviation of the data for the polymer that has different Tg values from different sources.


## Usage example for getting features and labels for the training/testing the model

```python
from polymetrix.data_loader import load_tg_dataset
from polymetrix.datasets import GlassTempDataset

df = load_tg_dataset('PolymerTg.csv')

dataset = GlassTempDataset(df=df)

all_data = len(dataset)

features = dataset.get_features(idx=range(all_data))
target = dataset.get_labels(idx=range(all_data))
```
This will output the array of features and labels for the dataset. which can be used for training/testing the model.





