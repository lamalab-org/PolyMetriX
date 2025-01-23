from typing import List, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors, AllChem
import logging


class BaseFeatureCalculator:
    agg_funcs = {
        "mean": np.mean,
        "min": np.min,
        "max": np.max,
        "sum": np.sum,
    }

    def __init__(self, agg: List[str] = ["sum"]):
        self.agg = agg

    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        raise NotImplementedError("Calculate method must be implemented by subclasses")

    def feature_base_labels(self) -> List[str]:
        raise NotImplementedError(
            "Feature labels method must be implemented by subclasses"
        )

    def feature_labels(self) -> List[str]:
        return [
            f"{label}_{agg}" for label in self.feature_base_labels() for agg in self.agg
        ]

    def aggregate(self, features: List[np.ndarray]) -> np.ndarray:
        results = []
        for agg_func in self.agg:
            if agg_func not in self.agg_funcs:
                raise ValueError(f"Unknown aggregation function: {agg_func}")
            results.append(self.agg_funcs[agg_func](features, axis=0))
        return np.concatenate(results)

    def get_feature_names(self) -> List[str]:
        raise NotImplementedError(
            "Get feature name method must be implemented by subclasses"
        )

    def citations(self) -> List[str]:
        return []

    def implementors(self) -> List[str]:
        return []


class NumHBondDonors(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        return np.array([Descriptors.NumHDonors(mol)])

    def feature_base_labels(self) -> List[str]:
        return ["num_hbond_donors"]


class NumHBondAcceptors(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        return np.array([Descriptors.NumHAcceptors(mol)])

    def feature_base_labels(self) -> List[str]:
        return ["num_hbond_acceptors"]


class NumRotatableBonds(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        return np.array([Descriptors.NumRotatableBonds(mol)])

    def feature_base_labels(self) -> List[str]:
        return ["num_rotatable_bonds"]


class NumRings(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        ring_info = mol.GetRingInfo()
        num_rings = len(ring_info.AtomRings())
        return np.array([num_rings])

    def feature_base_labels(self) -> List[str]:
        return ["num_rings"]


class NumAtoms(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        return np.array([mol.GetNumAtoms()])

    def feature_base_labels(self) -> List[str]:
        return ["num_atoms"]


class NumNonAromaticRings(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        non_aromatic_rings = sum(
            1
            for ring in mol.GetRingInfo().AtomRings()
            if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        )
        return np.array([non_aromatic_rings])

    def feature_base_labels(self) -> List[str]:
        return ["num_non_aromatic_rings"]


class NumAromaticRings(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        aromatic_rings = sum(
            1
            for ring in mol.GetRingInfo().AtomRings()
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        )
        return np.array([aromatic_rings])

    def feature_base_labels(self) -> List[str]:
        return ["num_aromatic_rings"]


class TopologicalSurfaceArea(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        return np.array([Descriptors.TPSA(mol)])

    def feature_base_labels(self) -> List[str]:
        return ["topological_surface_area"]


class FractionBicyclicRings(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        bicyclic_count = 0

        for i, ring1 in enumerate(atom_rings):
            for ring2 in atom_rings[i + 1 :]:
                if set(ring1).intersection(set(ring2)):
                    bicyclic_count += 1
                    break  # Count each bicyclic structure only once

        total_rings = len(atom_rings)
        fraction_bicyclic = bicyclic_count / total_rings if total_rings > 0 else 0
        return np.array([fraction_bicyclic])

    def feature_base_labels(self) -> List[str]:
        return ["fraction_bicyclic_rings"]


class NumAliphaticHeterocycles(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        num_heterocycles = 0
        for ring in mol.GetRingInfo().AtomRings():
            if any(mol.GetAtomWithIdx(atom).GetAtomicNum() != 6 for atom in ring):
                num_heterocycles += 1
        return np.array([num_heterocycles])

    def feature_base_labels(self) -> List[str]:
        return ["num_aliphatic_heterocycles"]


class SlogPVSA1(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        return np.array([Descriptors.SlogP_VSA1(mol)])

    def feature_base_labels(self) -> List[str]:
        return ["slogp_vsa1"]


class BalabanJIndex(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        return np.array([GraphDescriptors.BalabanJ(mol)])

    def feature_base_labels(self) -> List[str]:
        return ["balaban_j_index"]


class MolecularWeightFeaturizer(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        return np.array([Descriptors.ExactMolWt(mol)])

    def feature_base_labels(self) -> List[str]:
        return ["molecular_weight"]


class Sp3CarbonCountFeaturizer(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        sp3_count = sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetHybridization() == Chem.HybridizationType.SP3
        )
        return np.array([sp3_count])

    def feature_base_labels(self) -> List[str]:
        return ["sp3_carbon_count"]


class Sp2CarbonCountFeaturizer(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        sp2_count = sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetHybridization() == Chem.HybridizationType.SP2
        )
        return np.array([sp2_count])

    def feature_base_labels(self) -> List[str]:
        return ["sp2_carbon_count"]


class MaxEStateIndex(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        return np.array([Descriptors.MaxEStateIndex(mol)])

    def feature_base_labels(self) -> List[str]:
        return ["max_estate_index"]


class SMR_VSA5(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        return np.array([Descriptors.SMR_VSA5(mol)])

    def feature_base_labels(self) -> List[str]:
        return ["smr_vsa5"]


class FpDensityMorgan1(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        return np.array([Descriptors.FpDensityMorgan1(mol)])

    def feature_base_labels(self) -> List[str]:
        return ["fp_density_morgan1"]


class HeteroatomCount(BaseFeatureCalculator):
    @staticmethod
    def count_heteroatoms(mol: Chem.Mol) -> int:
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 6)

    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        return np.array([HeteroatomCount.count_heteroatoms(mol)])

    def feature_base_labels(self) -> List[str]:
        return ["heteroatom_count"]


class HeteroatomDensity(BaseFeatureCalculator):
    @staticmethod
    def count_heteroatoms(mol: Chem.Mol) -> int:
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 6)
    
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        num_atoms = mol.GetNumAtoms()
        num_heteroatoms = HeteroatomDensity.count_heteroatoms(mol)
        density = num_heteroatoms / num_atoms if num_atoms > 0 else 0
        return np.array([density])

    def feature_base_labels(self) -> List[str]:
        return ["heteroatom_density"]


class HeteroatomDistanceStats(BaseFeatureCalculator):
    def __init__(self, agg: List[str] = ["mean", "min", "max"]):
        super().__init__(agg)

    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        heteroatom_indices = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() not in [1, 6]
        ]

        if len(heteroatom_indices) < 2:
            return np.array([0] * len(self.agg))

        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)

        conf = mol.GetConformer()
        distances = []

        for i in range(len(heteroatom_indices)):
            for j in range(i + 1, len(heteroatom_indices)):
                pos1 = conf.GetAtomPosition(heteroatom_indices[i])
                pos2 = conf.GetAtomPosition(heteroatom_indices[j])
                distance = ((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2) ** 0.5
                distances.append(distance)

        if not distances:
            return np.array([0] * len(self.agg))

        return np.array([self.agg_funcs[agg](distances) for agg in self.agg])

    def feature_base_labels(self) -> List[str]:
        return [f"heteroatom_distance_{agg}" for agg in self.agg]


class HalogenCounts(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        halogen_counts = {9: 0, 17: 0, 35: 0, 53: 0}  # F, Cl, Br, I
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num in halogen_counts:
                halogen_counts[atomic_num] += 1

        total_halogens = sum(halogen_counts.values())
        return np.array(
            [total_halogens, halogen_counts[9], halogen_counts[17], halogen_counts[35]]
        )

    def feature_base_labels(self) -> List[str]:
        return ["total_halogens", "fluorine_count", "chlorine_count", "bromine_count"]


class BondCounts(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        single_bonds = 0
        double_bonds = 0
        triple_bonds = 0

        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            if bond_type == Chem.BondType.SINGLE:
                single_bonds += 1
            elif bond_type == Chem.BondType.DOUBLE:
                double_bonds += 1
            elif bond_type == Chem.BondType.TRIPLE:
                triple_bonds += 1

        return np.array([single_bonds, double_bonds, triple_bonds])

    def feature_base_labels(self) -> List[str]:
        return ["single_bonds", "double_bonds", "triple_bonds"]


class BridgingRingsCount(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
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


class MaxRingSize(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        Chem.SanitizeMol(mol)
        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()

        if not rings:
            return np.array([0])

        max_size = max(len(ring) for ring in rings)
        return np.array([max_size])

    def feature_base_labels(self) -> List[str]:
        return ["max_ring_size"]


class PolymerPartFeaturizer:
    def __init__(self, calculator: Optional[BaseFeatureCalculator] = None):
        self.calculator = calculator

    def featurize(self, polymer) -> np.ndarray:
        raise NotImplementedError("Featurize method must be implemented by subclasses")

    def feature_labels(self) -> List[str]:
        if self.calculator:
            return [
                f"{label}_{self.__class__.__name__.lower()}"
                for label in self.calculator.feature_base_labels()
            ]
        else:
            return [self.__class__.__name__.lower()]


class SideChainFeaturizer(PolymerPartFeaturizer):
    def featurize(self, polymer) -> np.ndarray:
        sidechain_mols = polymer.get_backbone_and_sidechain_molecules()[1]
        if not sidechain_mols:
            logging.info("No sidechains found in the molecule")
            return np.zeros(
                len(self.calculator.feature_base_labels()) * len(self.calculator.agg)
            )
        features = [
            self.calculator.calculate(mol).reshape(-1, 1) for mol in sidechain_mols
        ]
        return self.calculator.aggregate(np.concatenate(features))

    def feature_labels(self) -> List[str]:
        if self.calculator:
            return [
                f"{label}_{self.__class__.__name__.lower()}_{agg}"
                for label in self.calculator.feature_base_labels()
                for agg in self.calculator.agg
            ]
        else:
            return [self.__class__.__name__.lower()]


class NumSideChainFeaturizer(PolymerPartFeaturizer):
    def featurize(self, polymer) -> np.ndarray:
        sidechain_mols = polymer.get_backbone_and_sidechain_molecules()[1]
        return np.array([len(sidechain_mols)])


class BackBoneFeaturizer(PolymerPartFeaturizer):
    def featurize(self, polymer) -> np.ndarray:
        backbone_mol = polymer.get_backbone_and_sidechain_molecules()[0][0]
        return self.calculator.calculate(backbone_mol)


class NumBackBoneFeaturizer(PolymerPartFeaturizer):
    def featurize(self, polymer) -> np.ndarray:
        backbone_mols = polymer.get_backbone_and_sidechain_molecules()[0]
        return np.array([len(backbone_mols)])


class FullPolymerFeaturizer(PolymerPartFeaturizer):
    def featurize(self, polymer) -> np.ndarray:
        mol = Chem.MolFromSmiles(polymer.psmiles)
        return self.calculator.calculate(mol)


class MultipleFeaturizer:
    def __init__(self, featurizers: List[PolymerPartFeaturizer]):
        self.featurizers = featurizers

    def featurize(self, polymer) -> np.ndarray:
        features = []
        for featurizer in self.featurizers:
            feature = featurizer.featurize(polymer)
            if isinstance(feature, (int, float)):
                feature = np.array([feature])
            features.append(feature.flatten())
        return np.concatenate(features)

    def feature_labels(self) -> List[str]:
        labels = []
        for featurizer in self.featurizers:
            labels.extend(featurizer.feature_labels())
        return labels

    def citations(self) -> List[str]:
        citations = []
        for featurizer in self.featurizers:
            if hasattr(featurizer, "calculator") and featurizer.calculator:
                citations.extend(featurizer.calculator.citations())
        return list(set(citations))

    def implementors(self) -> List[str]:
        implementors = []
        for featurizer in self.featurizers:
            if hasattr(featurizer, "calculator") and featurizer.calculator:
                implementors.extend(featurizer.calculator.implementors())
        return list(set(implementors))