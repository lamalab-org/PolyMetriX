import pytest
from rdkit import Chem
import networkx as nx
from polymetrix.core.descriptors import mol_to_nx

@pytest.fixture
def sample_mol():
    return Chem.MolFromSmiles("*C(=O)C1=CC=C(C=C1)C(=O)NC1=CC=C(C=C1)N*")

@pytest.fixture
def sample_graph(sample_mol):
    return mol_to_nx(sample_mol)
