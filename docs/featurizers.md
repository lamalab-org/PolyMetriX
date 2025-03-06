# Features Available in PolyMetriX

The featurizers in PolyMetriX are classified into two categories:
**Chemical Featurizers**: These focus on capturing the chemical characteristics of polymers, such as types of atoms, functional groups, and chemical bonds, providing insights into how molecular composition influences behavior.
**Topological Featurizers**: These emphasize the structural and spatial arrangement of polymer components, assessing topology like connectivity and branching to understand their impact on material performance.

## Featurizers Overview

Below is a detailed table of the featurizers available in PolyMetriX:

| Featurizer Name                       | Description                                                                                   | Type of Featurizer     |
|:--------------------------------------|:---------------------------------------------------------------------------------------------|:------------:|
| **NumHBondDonors**                   | Counts hydrogen bond donors, indicating ability to form hydrogen bonds with water.           | Chemical    |
| **NumHBondAcceptors**                | Counts hydrogen bond acceptors, reflecting interaction potential with water.                 | Chemical    |
| **NumRotatableBonds**                | Counts rotatable bonds, providing insights into flexibility and conformational freedom.      | Chemical    |
| **NumRings**                         | Counts total rings, affecting structural stability and molecular interactions.               | Chemical    |
| **NumNonAromaticRings**              | Counts non-aromatic rings in the polymer structure.                                          | Chemical    |
| **NumAromaticRings**                 | Counts aromatic rings, influencing stability and reactivity.                                 | Chemical    |
| **NumAtoms**                         | Counts total atoms in the polymer.                                                           | Chemical    |
| **TopologicalSurfaceArea**           | Measures polar surface area, affecting interactions with polar solvents like water.          | Chemical    |
| **FractionBicyclicRings**            | Fraction of bicyclic rings, impacting rigidity and thermal stability.                        | Chemical    |
| **NumAliphaticHeterocycles**         | Counts non-aromatic heterocycles with heteroatoms (e.g., N, O, S).                           | Chemical    |
| **SlogPVSA1**                        | Surface area contributing to octanol solubility, linked to lipophilicity.                    | Chemical    |
| **BalabanJIndex**                    | Measures molecular complexity and connectivity of atoms.                                     | Chemical    |
| **MolecularWeightFeaturizer**        | Calculates molecular weight, influencing solubility and other properties.                    | Chemical    |
| **Sp3CarbonCountFeaturizer**         | Counts sp3 carbons, providing info on 3D structure and solubility.                           | Chemical    |
| **Sp2CarbonCountFeaturizer**         | Counts sp2 carbons, indicating aromaticity and reactivity.                                   | Chemical    |
| **MaxEStateIndex**                   | Maximum electronic state index, reflecting charge distribution.                              | Chemical    |
| **SMR_VSA5**                         | Molar refractivity sum for atoms with specific surface area (2.45–2.75).                     | Chemical    |
| **FpDensityMorgan1**                 | Density of substructure info in Morgan fingerprint.                                          | Chemical    |
| **HalogenCounts**                    | Counts halogen atoms (F, Cl, Br, I) in the molecule.                                         | Chemical    |
| **BondCounts**                       | Counts total bonds, indicating structural complexity and reactivity.                         | Chemical    |
| **BridgingRingsCount**               | Counts bridging rings, affecting structural stability and rigidity.                          | Chemical    |
| **MaxRingSize**                      | Calculates the largest ring size in the molecule.                                            | Chemical    |
| **HeteroatomCount**                  | Counts heteroatoms (non-C, non-H) in heterocyclic rings.                                     | Chemical    |
| **HeteroatomDensity**                | Density of heteroatoms in the molecule.                                                      | Chemical    |
| **HeteroatomDistanceStats**          | Statistics on distances between heteroatoms.                                                 | Chemical    |
| **NumSideChainFeaturizer**           | Counts sidechains, influencing crystallinity and density.                                    | Topological |
| **NumBackBoneFeaturizer**            | Counts backbone atoms in the polymer.                                                        | Topological |
| **SideChainLengthFeaturizer**        | Measures sidechain length in the polymer.                                                    | Topological |
| **BackBoneLengthFeaturizer**         | Measures backbone length in the polymer.                                                     | Topological |
| **SidechainLengthToStarAttachmentDistanceRatioFeaturizer** | Ratio of sidechain length to minimum distance from star nodes.                | Topological |
| **StarToSidechainMinDistanceFeaturizer** | Minimum distance from star nodes to sidechains in edges.                         | Topological |
| **SidechainDiversityFeaturizer**     | Counts structurally diverse sidechains using Weisfeiler-Lehman graph hash.                   | Topological |
