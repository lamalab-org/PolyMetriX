import pytest
import networkx as nx
from rdkit import Chem
from polymetrix.polymer import (
    Polymer,
)  # Assuming the Polymer class is in a file named polymer.py


def test_polymer_creation():
    polymer = Polymer.from_psmiles("*C1CCC(*)C1")
    assert isinstance(polymer, Polymer)
    assert polymer.psmiles == "*C1CCC(*)C1"
    assert isinstance(polymer.graph, nx.Graph)


def test_invalid_psmiles():
    with pytest.raises(ValueError):
        Polymer.from_psmiles("invalid_smiles")


def test_backbone_and_sidechain_identification():
    polymer = Polymer.from_psmiles("*C1CCC(CC)(*)C1")
    assert len(set(polymer.backbone_nodes)) == 7
    assert len(set(polymer.sidechain_nodes)) == 2


def test_complex_polymer_backbone_and_sidechain():
    polymer = Polymer.from_psmiles("*C1C(CC)C(C(C)C)CC(*)C1")
    assert len(set(polymer.backbone_nodes)) == 8
    assert len(set(polymer.sidechain_nodes)) == 5


def test_get_backbone_and_sidechain_molecules():
    polymer = Polymer.from_psmiles("*C1CCC(CC)(*)C1")
    backbone, sidechains = polymer.get_backbone_and_sidechain_molecules()
    assert len(backbone) == 1
    assert len(sidechains) == 1
    assert isinstance(backbone[0], Chem.Mol)
    assert isinstance(sidechains[0], Chem.Mol)


def test_get_backbone_and_sidechain_graphs():
    polymer = Polymer.from_psmiles("*C1CCC(CC)(*)C1")
    backbones, sidechains = polymer.get_backbone_and_sidechain_graphs()
    assert isinstance(backbones[0], nx.Graph)
    assert len(sidechains) == 1
    assert isinstance(sidechains[0], nx.Graph)


def test_multiple_sidechains():
    polymer = Polymer.from_psmiles('*c1ccc(Oc2ccc(-c3nc4cc(Oc5ccc(NC(=O)c6cc(C(=O)Nc7ccc(Oc8ccc9nc(-c%10ccccc%10)c(*)nc9c8)cc7)cc(N7C(=O)c8ccccc8C7=O)c6)cc5)ccc4nc3-c3ccccc3)cc2)cc1')
    backbones, sidechains = polymer.get_backbone_and_sidechain_graphs()
    assert len(backbones) == 1
    assert len(sidechains) == 3

    backbone_molecules, sidechains_molecules = polymer.get_backbone_and_sidechain_molecules()
    assert len(backbone_molecules) == 1
    assert len(sidechains_molecules) == 3

    original_number_of_atoms = len(polymer.graph)
    sidechain_smiles = [
        Chem.MolToSmiles(sidechain_mol) for sidechain_mol in sidechains_molecules
    ]
    assert set(sidechain_smiles) == {'c1ccccc1', 'O=C1NC(=O)c2ccccc21'}
    assert original_number_of_atoms == len(backbones[0]) + sum([len(sidechain) for sidechain in sidechains])


def test_calculate_molecular_weight():
    polymer = Polymer.from_psmiles("*C1CCC(*)C1")
    mw = polymer.calculate_molecular_weight()
    assert isinstance(mw, float)
    assert pytest.approx(mw, abs=0.1) == 68.062600256


def test_psmiles_setter():
    polymer = Polymer()
    polymer.psmiles = "*C1CCC(*)C1"
    assert polymer.psmiles == "*C1CCC(*)C1"
    assert isinstance(polymer.graph, nx.Graph)
    assert polymer.backbone_nodes is not None
    assert polymer.sidechain_nodes is not None


def test_graph_property():
    polymer = Polymer.from_psmiles("*C1CCC(*)C1")
    graph = polymer.graph
    assert isinstance(graph, nx.Graph)
    assert len(graph.nodes) == 7
    assert len(graph.edges) == 7


def test_connection_points():
    polymer = Polymer.from_psmiles("*C1CCC(*)C1")
    connection_points = polymer.get_connection_points()
    assert len(connection_points) == 2
    assert all(polymer.graph.nodes[cp]["element"] == "*" for cp in connection_points)
