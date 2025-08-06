from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from polymetrix.featurizers.base_featurizer import MoleculeFeaturizer


class Molecule:
    """A class to represent a general molecule from SMILES string.

    Attributes:
        smiles: Optional[str], the SMILES string representing the molecule.
        mol: Optional[Chem.Mol], the RDKit molecule object.

    Raises:
        ValueError: If the provided SMILES string is invalid or cannot be processed.
    """

    def __init__(self):
        self._smiles: Optional[str] = None
        self._mol: Optional[Chem.Mol] = None

    @classmethod
    def from_smiles(cls, smiles: str) -> "Molecule":
        """Creates a Molecule instance from a SMILES string.

        Args:
            smiles: str, the SMILES string representing the molecule.

        Returns:
            Molecule: A new Molecule object initialized with the given SMILES string.

        Raises:
            ValueError: If the SMILES string is invalid.
        """
        molecule = cls()
        molecule.smiles = smiles
        return molecule

    @property
    def smiles(self) -> Optional[str]:
        """Gets the SMILES string of the molecule.

        Returns:
            Optional[str]: The SMILES string, or None if not set.
        """
        return self._smiles

    @smiles.setter
    def smiles(self, value: str):
        """Sets the SMILES string and creates the RDKit molecule object.

        Args:
            value: str, the SMILES string to set.

        Raises:
            ValueError: If the SMILES string is invalid, cannot be processed, or contains polymer connection points (*).
        """
        try:
            # Check for asterisk atoms which indicate pSMILES (polymer SMILES)
            if "*" in value:
                raise ValueError(
                    "SMILES string contains asterisk atoms (*) which indicates a pSMILES string. Use Polymer.from_psmiles() instead for polymer molecules."
                )

            mol = Chem.MolFromSmiles(value)
            if mol is None:
                raise ValueError("Invalid SMILES string")
            self._smiles = value
            self._mol = mol
        except Exception as e:
            raise ValueError(f"Error processing SMILES: {str(e)}") from e

    @property
    def mol(self) -> Optional[Chem.Mol]:
        """Gets the RDKit molecule object.

        Returns:
            Optional[Chem.Mol]: The RDKit molecule object, or None if not set.
        """
        return self._mol

    def calculate_molecular_weight(self) -> float:
        """Calculates the exact molecular weight of the molecule.

        Returns:
            float: The molecular weight of the molecule.
        """
        if self._mol is None:
            raise ValueError("No molecule set")
        return ExactMolWt(self._mol)


class FullMolecularFeaturizer(MoleculeFeaturizer):
    """Featurizer for general molecules.

    This class can featurize any molecule from a Molecule object that contains
    a SMILES string and RDKit molecule object.
    """

    def featurize(self, molecule) -> np.ndarray:
        """Featurize a molecule object.

        Args:
            molecule: A Molecule object with a mol property containing an RDKit molecule.

        Returns:
            np.ndarray: Feature vector calculated by the underlying calculator.
        """
        if molecule.mol is None:
            raise ValueError("Molecule object does not contain a valid RDKit molecule")
        return self.calculator.calculate(molecule.mol)
