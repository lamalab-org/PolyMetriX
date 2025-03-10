from typing import List
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors
from .base_featurizer import BaseFeatureCalculator


class GenericScalarFeaturizer(BaseFeatureCalculator):
    def __init__(
        self, func, label: str, sanitize_default: bool = True, agg: List[str] = None
    ):
        super().__init__(agg)
        self.func = func
        self.label = label
        self.sanitize_default = sanitize_default
        self.agg = agg if agg is not None else ["sum"]

    def calculate(self, mol: Chem.Mol, sanitize: bool = None) -> np.ndarray:
        if sanitize is None:
            sanitize = self.sanitize_default
        self._sanitize(mol, sanitize)
        return np.array([self.func(mol)])

    def feature_base_labels(self) -> List[str]:
        return [self.label]


class NumHBondDonors(GenericScalarFeaturizer):
    """
    Counts Number of hydrogen bond donors.
    """

    def __init__(self, agg: List[str] = None):
        super().__init__(Descriptors.NumHDonors, "num_hbond_donors", agg=agg)


class NumHBondAcceptors(GenericScalarFeaturizer):
    """
    Counts Number of hydrogen bond acceptors.
    """

    def __init__(self, agg: List[str] = None):
        super().__init__(Descriptors.NumHAcceptors, "num_hbond_acceptors", agg=agg)


class NumRotatableBonds(GenericScalarFeaturizer):
    """
    Counts Number of rotatable bonds.
    """

    def __init__(self, agg: List[str] = None):
        super().__init__(Descriptors.NumRotatableBonds, "num_rotatable_bonds", agg=agg)


class TopologicalSurfaceArea(GenericScalarFeaturizer):
    """
    Calculates the topological polar surface area.
    """

    def __init__(self, agg: List[str] = None):
        super().__init__(Descriptors.TPSA, "topological_surface_area", agg=agg)


class SlogPVSA1(GenericScalarFeaturizer):
    """
    Calculates the Surface area contributing to octanol solubility, linked to lipophilicity.
    """

    def __init__(self, agg: List[str] = None):
        super().__init__(Descriptors.SlogP_VSA1, "slogp_vsa1", agg=agg)


class BalabanJIndex(GenericScalarFeaturizer):
    """
    Measures molecular complexity and connectivity of atoms.
    """

    def __init__(self, agg: List[str] = None):
        super().__init__(GraphDescriptors.BalabanJ, "balaban_j_index", agg=agg)


class MolecularWeight(GenericScalarFeaturizer):
    """
    Calculates the molecular weight of the molecule.
    """

    def __init__(self, agg: List[str] = None):
        super().__init__(Descriptors.ExactMolWt, "molecular_weight", agg=agg)


class MaxEStateIndex(GenericScalarFeaturizer):
    """
    Maximum electronic state index, reflecting charge distribution.
    """

    def __init__(self, agg: List[str] = None):
        super().__init__(Descriptors.MaxEStateIndex, "max_estate_index", agg=agg)


class SmrVSA5(GenericScalarFeaturizer):
    """
    Molar refractivity sum for atoms with specific surface area (2.45–2.75).
    """

    def __init__(self, agg: List[str] = None):
        super().__init__(Descriptors.SMR_VSA5, "smr_vsa5", agg=agg)


class FpDensityMorgan1(GenericScalarFeaturizer):
    """
    Calculates the density of the Morgan1 fingerprint.
    """

    def __init__(self, agg: List[str] = None):
        super().__init__(Descriptors.FpDensityMorgan1, "fp_density_morgan1", agg=agg)


class NumRings(BaseFeatureCalculator):
    """
    Counts the number of rings in the molecule.
    """

    def calculate(self, mol: Chem.Mol, sanitize: bool = True) -> np.ndarray:
        self._sanitize(mol, sanitize)
        return np.array([len(mol.GetRingInfo().AtomRings())])

    def feature_base_labels(self) -> List[str]:
        return ["num_rings"]


class NumAtoms(BaseFeatureCalculator):
    """
    Counts the number of atoms in the molecule.
    """

    def calculate(self, mol: Chem.Mol, sanitize: bool = False) -> np.ndarray:
        return np.array([mol.GetNumAtoms()])

    def feature_base_labels(self) -> List[str]:
        return ["num_atoms"]


class NumNonAromaticRings(BaseFeatureCalculator):
    """
    Counts the number of non-aromatic rings in the molecule.
    """

    def calculate(self, mol: Chem.Mol, sanitize: bool = True) -> np.ndarray:
        self._sanitize(mol, sanitize)
        return np.array(
            [
                sum(
                    1
                    for ring in mol.GetRingInfo().AtomRings()
                    if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
                )
            ]
        )

    def feature_base_labels(self) -> List[str]:
        return ["num_non_aromatic_rings"]


class NumAliphaticHeterocycles(BaseFeatureCalculator):
    """
    Counts the number of aliphatic heterocycles in the molecule.
    """

    def calculate(self, mol: Chem.Mol, sanitize: bool = True) -> np.ndarray:
        self._sanitize(mol, sanitize)
        num_heterocycles = 0
        for ring in mol.GetRingInfo().AtomRings():
            if any(mol.GetAtomWithIdx(atom).GetAtomicNum() != 6 for atom in ring):
                num_heterocycles += 1
        return np.array([num_heterocycles])

    def feature_base_labels(self) -> List[str]:
        return ["num_aliphatic_heterocycles"]


class NumAromaticRings(BaseFeatureCalculator):
    """
    Counts the number of aromatic rings in the molecule.
    """

    def calculate(self, mol: Chem.Mol, sanitize: bool = True) -> np.ndarray:
        self._sanitize(mol, sanitize)
        return np.array(
            [
                sum(
                    1
                    for ring in mol.GetRingInfo().AtomRings()
                    if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
                )
            ]
        )

    def feature_base_labels(self) -> List[str]:
        return ["num_aromatic_rings"]


class FractionBicyclicRings(BaseFeatureCalculator):
    """
    Calculates the fraction of bicyclic rings in the molecule.
    """

    def calculate(self, mol: Chem.Mol, sanitize: bool = True) -> np.ndarray:
        self._sanitize(mol, sanitize)
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        bicyclic_count = sum(
            1
            for i, ring1 in enumerate(atom_rings)
            for ring2 in atom_rings[i + 1 :]
            if set(ring1) & set(ring2)
        )
        return np.array([bicyclic_count / len(atom_rings) if atom_rings else 0])

    def feature_base_labels(self) -> List[str]:
        return ["fraction_bicyclic_rings"]


class HeteroatomCount(BaseFeatureCalculator):
    """
    Counts heteroatoms (non-C, non-H) in heterocyclic rings.
    """

    def calculate(self, mol: Chem.Mol, sanitize: bool = True) -> np.ndarray:
        self._sanitize(mol, sanitize)
        return np.array([sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 6)])

    def feature_base_labels(self) -> List[str]:
        return ["heteroatom_count"]


class HeteroatomDensity(BaseFeatureCalculator):
    """
    Density of heteroatoms in the molecule.
    """

    def calculate(self, mol: Chem.Mol, sanitize: bool = True) -> np.ndarray:
        self._sanitize(mol, sanitize)
        num_atoms = mol.GetNumAtoms()
        num_heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 6)
        return np.array([num_heteroatoms / num_atoms if num_atoms else 0])

    def feature_base_labels(self) -> List[str]:
        return ["heteroatom_density"]


class BondCounts(BaseFeatureCalculator):
    """
    Counts the number of single, double, and triple bonds in the molecule.
    """

    def calculate(self, mol: Chem.Mol, sanitize: bool = True) -> np.ndarray:
        self._sanitize(mol, sanitize)
        bond_types = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 0,
            Chem.BondType.TRIPLE: 0,
        }
        for bond in mol.GetBonds():
            if bond.GetBondType() in bond_types:
                bond_types[bond.GetBondType()] += 1
        return np.array(
            [
                bond_types[Chem.BondType.SINGLE],
                bond_types[Chem.BondType.DOUBLE],
                bond_types[Chem.BondType.TRIPLE],
            ]
        )

    def feature_base_labels(self) -> List[str]:
        return ["single_bonds", "double_bonds", "triple_bonds"]


class MaxRingSize(BaseFeatureCalculator):
    """
    Calculates the size of the largest ring in the molecule.
    """

    def calculate(self, mol: Chem.Mol, sanitize: bool = True) -> np.ndarray:
        self._sanitize(mol, sanitize)
        rings = mol.GetRingInfo().AtomRings()
        return np.array([max(map(len, rings)) if rings else 0])

    def feature_base_labels(self) -> List[str]:
        return ["max_ring_size"]


class Sp3CarbonCountFeaturizer(BaseFeatureCalculator):
    """
    Counts the number of sp3 hybridized carbon atoms in the molecule.
    """

    def calculate(self, mol: Chem.Mol, sanitize: bool = True) -> np.ndarray:
        self._sanitize(mol, sanitize)
        sp3_count = sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetHybridization() == Chem.HybridizationType.SP3
        )
        return np.array([sp3_count])

    def feature_base_labels(self) -> List[str]:
        return ["sp3_carbon_count"]


class Sp2CarbonCountFeaturizer(BaseFeatureCalculator):
    """
    Counts the number of sp2 hybridized carbon atoms in the molecule.
    """

    def calculate(self, mol: Chem.Mol, sanitize: bool = True) -> np.ndarray:
        self._sanitize(mol, sanitize)
        sp2_count = sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetHybridization() == Chem.HybridizationType.SP2
        )
        return np.array([sp2_count])

    def feature_base_labels(self) -> List[str]:
        return ["sp2_carbon_count"]


class HalogenCounts(BaseFeatureCalculator):
    """
    Counts the number of halogen atoms in the molecule.
    """

    def calculate(self, mol: Chem.Mol, sanitize: bool = True) -> np.ndarray:
        self._sanitize(mol, sanitize)
        halogen_counts = {9: 0, 17: 0, 35: 0, 53: 0}  # F, Cl, Br, I
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num in halogen_counts:
                halogen_counts[atomic_num] += 1

        total_halogens = sum(halogen_counts.values())

        return np.array(
            [
                total_halogens,
                halogen_counts[9],
                halogen_counts[17],
                halogen_counts[35],
                halogen_counts[53],
            ]
        )

    def feature_base_labels(self) -> List[str]:
        return [
            "total_halogens",
            "fluorine_count",
            "chlorine_count",
            "bromine_count",
            "iodine_count",
        ]


class BridgingRingsCount(BaseFeatureCalculator):
    """
    Counts the number of bridging rings in the molecule.
    """

    def calculate(self, mol: Chem.Mol, sanitize: bool = True) -> np.ndarray:
        self._sanitize(mol, sanitize)
        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()
        bridging_rings = 0

        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                if len(set(rings[i]) & set(rings[j])) >= 2:
                    bridging_rings += 1
                    break

        return np.array([bridging_rings])

    def feature_base_labels(self) -> List[str]:
        return ["bridging_rings_count"]
