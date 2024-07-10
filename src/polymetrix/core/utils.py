from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np
# import wandb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def get_fp_ecfp_bitvector(smiles, radius=2, nBits=2048, info={}):
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
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits, bitInfo=info)
    else:
        return None
    
def calculate_metrics(actual, predicted, model_name, prefix=""):
    """
    Calculate evaluation metrics for model predictions.

    Args:
        actual (np.ndarray): Array of actual target values.
        predicted (np.ndarray): Array of predicted target values.
        model_name (str): Name of the model.
        prefix (str, optional): Prefix for metric names in logging. Defaults to "".

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    metrics = {
        'model_name': model_name,
        'r2': r2_score(actual, predicted),
        'mae': mean_absolute_error(actual, predicted),
        'mse': mean_squared_error(actual, predicted),
        'rmse': np.sqrt(mean_squared_error(actual, predicted)),
        'mape': np.mean(np.abs((actual - predicted) / actual)) * 100
    }
    # for key, value in metrics.items():
    #     wandb.log({f"{prefix}_{key}": value})
    return metrics