from typing import List, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


class BaseFeatureCalculator:
    def __init__(self, agg: List[str] = ["sum"]):
        self.agg = agg

    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        raise NotImplementedError(
            "Calculate method must be implemented by subclasses")

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
            if agg_func == "sum":
                results.append(np.sum(features, axis=0))
            elif agg_func == "mean":
                results.append(np.mean(features, axis=0))
            elif agg_func == "min":
                results.append(np.min(features, axis=0))
            elif agg_func == "max":
                results.append(np.max(features, axis=0))
            else:
                raise ValueError(f"Unknown aggregation function: {agg_func}")
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
        return np.array([Descriptors.NumHDonors(mol)])

    def feature_base_labels(self) -> List[str]:
        return ["num_hbond_donors"]


class NumHBondAcceptors(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        return np.array([Descriptors.NumHAcceptors(mol)])

    def feature_base_labels(self) -> List[str]:
        return ["num_hbond_acceptors"]


class NumRotatableBonds(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        return np.array([Descriptors.NumRotatableBonds(mol)])

    def feature_base_labels(self) -> List[str]:
        return ["num_rotatable_bonds"]


class NumRings(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        ring_info = mol.GetRingInfo()
        num_rings = len(ring_info.AtomRings())
        return np.array([num_rings])

    def feature_base_labels(self) -> List[str]:
        return ["num_rings"]

class NumAtoms(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        return np.array(mol.GetNumAtoms())

    def feature_base_labels(self) -> List[str]:
        return ['num_atoms']

class NumNonAromaticRings(BaseFeatureCalculator):
    def calculate(self, mol: Chem.Mol) -> np.ndarray:
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
        aromatic_rings = sum(
            1
            for ring in mol.GetRingInfo().AtomRings()
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        )
        return np.array([aromatic_rings])

    def feature_base_labels(self) -> List[str]:
        return ["num_aromatic_rings"]


class PolymerPartFeaturizer:
    def __init__(self, calculator: Optional[BaseFeatureCalculator] = None):
        self.calculator = calculator

    def featurize(self, polymer) -> np.ndarray:
        raise NotImplementedError(
            "Featurize method must be implemented by subclasses")

    def feature_labels(self) -> List[str]:
        return [
            f"{label}_{self.__class__.__name__.lower()}"
            for label in self.calculator.feature_labels()
        ]


class SideChainFeaturizer(PolymerPartFeaturizer):
    def featurize(self, polymer) -> np.ndarray:
        sidechain_mols = polymer.get_backbone_and_sidechain_molecules()[1]
        features = [self.calculator.calculate(mol).reshape(-1,1) for mol in sidechain_mols]
        return self.calculator.aggregate(np.concatenate(features))

class NumSideChainFeaturizer(PolymerPartFeaturizer):
    def featurize(self, polymer) -> np.ndarray:
        sidechain_mols = polymer.get_backbone_and_sidechain_molecules()[1]
        return np.array([len(sidechain_mols)])

class BackBoneFeaturizer(PolymerPartFeaturizer):
    def featurize(self, polymer) -> np.ndarray:
        backbone_mol = polymer.get_backbone_and_sidechain_molecules()[0][0]
        return self.calculator.calculate(backbone_mol)


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
            features.append(featurizer.featurize(polymer))
        return np.concatenate(features)

    def feature_labels(self) -> List[str]:
        labels = []
        for featurizer in self.featurizers:
            labels.extend(featurizer.feature_labels())
        return labels

    def citations(self) -> List[str]:
        citations = []
        for featurizer in self.featurizers:
            citations.extend(featurizer.calculator.citations())
        return list(set(citations))

    def implementors(self) -> List[str]:
        implementors = []
        for featurizer in self.featurizers:
            implementors.extend(featurizer.calculator.implementors())
        return list(set(implementors))
