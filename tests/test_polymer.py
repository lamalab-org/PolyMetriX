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
