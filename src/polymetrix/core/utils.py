import logging
import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_fp_ecfp_bitvector(smiles, radius=2, nBits=2048, info=None):
    """
    Generate ECFP fingerprint bit vector for a given SMILES string.

    Args:
        smiles (str): SMILES string.
        radius (int, optional): Fingerprint radius. Defaults to 2.
        nBits (int, optional): Number of bits in the fingerprint. Defaults to 2048.
        info (dict, optional): Dictionary to store bit info. Defaults to {}.

    Returns:
        rdkit.DataStructs.cDataStructs.ExplicitBitVect: ECFP fingerprint bit vector.
    """
    if info is None:
        info = {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits, bitInfo=info)
    return None


def make_linearpolymer(smiles, n=2, terminal="C"):
    """
    poly.make_linearpolymer

    Generate linearpolymer SMILES from monomer SMILES

    Args:
        smiles: SMILES (str)
        n: Polimerization degree (int)
        terminal: SMILES of terminal substruct (str)

    Returns:
        SMILES
    """

    dummy = "[*]"
    dummy_head = "[Nh]"
    dummy_tail = "[Og]"

    smiles_in = smiles
    smiles = smiles.replace("[*]", "*")
    smiles = smiles.replace("[3H]", "*")

    if smiles.count("*") != 2:
        logging.error("Error in make_linearpolymer: %s", smiles_in)
        return None

    smiles = smiles.replace("\\", "\\\\")
    smiles_head = re.sub(pattern=r"\*", repl=dummy_head, string=smiles, count=1)
    smiles_tail = re.sub(pattern=r"\*", repl=dummy_tail, string=smiles, count=1)
    # smiles_tail = re.sub(r'%s\\\\' % dummy_tail, '%s/' % dummy_tail, smiles_tail, 1)
    # smiles_tail = re.sub(r'%s/' % dummy_tail, '%s\\\\' % dummy_tail, smiles_tail, 1)

    try:
        mol_head = Chem.MolFromSmiles(smiles_head)
        mol_tail = Chem.MolFromSmiles(smiles_tail)
        mol_terminal = Chem.MolFromSmiles(terminal)
        mol_dummy = Chem.MolFromSmiles(dummy)
        mol_dummy_tail = Chem.MolFromSmiles(dummy_tail)

        con_point = 1
        for atom in mol_tail.GetAtoms():
            if atom.GetSymbol() == mol_dummy_tail.GetAtomWithIdx(0).GetSymbol():
                con_point = atom.GetNeighbors()[0].GetIdx()
                break

        for _poly in range(n - 1):
            mol_head = Chem.rdmolops.ReplaceSubstructs(
                mol_head, mol_dummy, mol_tail, replacementConnectionPoint=con_point
            )[0]
            mol_head = Chem.RWMol(mol_head)
            for atom in mol_head.GetAtoms():
                if atom.GetSymbol() == mol_dummy_tail.GetAtomWithIdx(0).GetSymbol():
                    idx = atom.GetIdx()
                    break
            mol_head.RemoveAtom(idx)
            Chem.SanitizeMol(mol_head)

        mol = mol_head.GetMol()
        mol = Chem.rdmolops.ReplaceSubstructs(
            mol, mol_dummy, mol_terminal, replacementConnectionPoint=0
        )[0]

        poly_smiles = Chem.MolToSmiles(mol)
        poly_smiles = poly_smiles.replace(dummy_head, terminal)

    except Exception as e:
        logging.error("Error in make_linearpolymer: %s", e)
        return None

    return poly_smiles


def calculate_metrics(actual, predicted, model_name):
    """
    Calculate evaluation metrics for model predictions.

    Args:
        actual (np.ndarray): Array of actual target values.
        predicted (np.ndarray): Array of predicted target values.
        model_name (str): Name of the model.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    return {
        "model_name": model_name,
        "r2": r2_score(actual, predicted),
        "mae": mean_absolute_error(actual, predicted),
        "mse": mean_squared_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "mape": np.mean(np.abs((actual - predicted) / actual)) * 100,
    }
