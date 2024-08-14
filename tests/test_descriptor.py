from rdkit import Chem
import networkx as nx
from polymetrix.core.descriptors import (
    mol_to_nx, find_shortest_paths_between_stars,
    find_cycles_including_paths, add_degree_one_nodes_to_backbone,
    classify_backbone_and_sidechains, get_real_backbone_and_sidechain_bridges,
    number_and_length_of_sidechains_and_backbones
)



def test_mol_to_nx(sample_mol):
    graph = mol_to_nx(sample_mol)
    assert isinstance(graph, nx.Graph)
    assert len(graph.nodes) == sample_mol.GetNumAtoms()
    assert len(graph.edges) == sample_mol.GetNumBonds()

def test_find_shortest_paths_between_stars(sample_graph):
    paths = find_shortest_paths_between_stars(sample_graph)
    assert len(paths) == 1
    assert len(paths[0]) > 0

def test_find_cycles_including_paths(sample_graph):
    paths = find_shortest_paths_between_stars(sample_graph)
    cycles = find_cycles_including_paths(sample_graph, paths)
    assert len(cycles) > 0

def test_add_degree_one_nodes_to_backbone(sample_graph):
    backbone = [0, 1, 2, 3]
    updated_backbone = add_degree_one_nodes_to_backbone(sample_graph, backbone)
    assert len(updated_backbone) >= len(backbone)

def test_classify_backbone_and_sidechains(sample_graph):
    backbone, sidechain = classify_backbone_and_sidechains(sample_graph)
    assert len(backbone) + len(sidechain) == len(sample_graph.nodes)

def test_get_real_backbone_and_sidechain_bridges(sample_graph):
    backbone, sidechain = classify_backbone_and_sidechains(sample_graph)
    bb_bridges, sc_bridges = get_real_backbone_and_sidechain_bridges(sample_graph, backbone, sidechain)
    assert len(bb_bridges) + len(sc_bridges) > 0

def test_number_and_length_of_sidechains_and_backbones():
    sc_bridges = [(0, 1), (1, 2), (2, 3)]
    bb_bridges = [(4, 5), (5, 6), (6, 7)]
    sidechains, backbones = number_and_length_of_sidechains_and_backbones(sc_bridges, bb_bridges)
    assert len(sidechains) == 1
    assert len(backbones) == 1
