from collections import OrderedDict
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt


class Polymer:
    def __init__(self):
        self._psmiles: Optional[str] = None
        self._graph: Optional[nx.Graph] = None
        self._backbone_nodes: Optional[List[int]] = None
        self._sidechain_nodes: Optional[List[int]] = None
        self._connection_points: Optional[List[int]] = None

    @classmethod
    def from_psmiles(cls, psmiles: str) -> "Polymer":
        polymer = cls()
        polymer.psmiles = psmiles
        return polymer

    @property
    def psmiles(self) -> Optional[str]:
        return self._psmiles

    @psmiles.setter
    def psmiles(self, value: str):
        try:
            mol = Chem.MolFromSmiles(value)
            if mol is None:
                raise ValueError("Invalid pSMILES string")
            self._psmiles = value
            self._graph = self._mol_to_nx(mol)
            self._identify_connection_points()
            self._identify_backbone_and_sidechain()
        except Exception as e:
            raise ValueError(f"Error processing pSMILES: {str(e)}") from e

    def _mol_to_nx(self, mol: Chem.Mol) -> nx.Graph:
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(
                atom.GetIdx(), atomic_num=atom.GetAtomicNum(), element=atom.GetSymbol()
            )
        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond_type=bond.GetBondType(),
            )
        return G

    def _identify_connection_points(self):
        self._connection_points = [
            node
            for node, data in self._graph.nodes(data=True)
            if data["element"] == "*"
        ]

    def _identify_backbone_and_sidechain(self):
        self._backbone_nodes, self._sidechain_nodes = classify_backbone_and_sidechains(
            self._graph
        )

    @property
    def backbone_nodes(self) -> List[int]:
        return self._backbone_nodes

    @property
    def sidechain_nodes(self) -> List[int]:
        return self._sidechain_nodes

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    def get_backbone_and_sidechain_molecules(
        self,
    ) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
        backbone_mol = self._subgraph_to_mol(
            self._graph.subgraph(self._backbone_nodes))
        sidechain_mols = [
            self._subgraph_to_mol(self._graph.subgraph(nodes))
            for nodes in nx.connected_components(
                self._graph.subgraph(self._sidechain_nodes)
            )
        ]
        return [backbone_mol], sidechain_mols

    def get_backbone_and_sidechain_graphs(self) -> Tuple[nx.Graph, List[nx.Graph]]:
        backbone_graph = self._graph.subgraph(self._backbone_nodes)
        sidechain_graphs = [
            self._graph.subgraph(nodes)
            for nodes in nx.connected_components(
                self._graph.subgraph(self._sidechain_nodes)
            )
        ]
        return [backbone_graph], sidechain_graphs

    def _subgraph_to_mol(self, subgraph: nx.Graph) -> Chem.Mol:
        mol = Chem.RWMol()
        node_to_idx = {}
        for node in subgraph.nodes():
            atom = Chem.Atom(subgraph.nodes[node]["atomic_num"])
            idx = mol.AddAtom(atom)
            node_to_idx[node] = idx
        for u, v, data in subgraph.edges(data=True):
            mol.AddBond(node_to_idx[u], node_to_idx[v], data["bond_type"])
        return mol.GetMol()

    def calculate_molecular_weight(self) -> float:
        mol = Chem.MolFromSmiles(self._psmiles)
        return ExactMolWt(mol)

    def get_connection_points(self) -> List[int]:
        return self._connection_points

    def number_and_length_of_sidechains_and_backbones(self):
        backbone_bridges, sidechain_bridges = get_real_backbone_and_sidechain_bridges(
            self._graph, self._backbone_nodes, self._sidechain_nodes
        )
        sidechains, backbones = number_and_length_of_sidechains_and_backbones(
            sidechain_bridges, backbone_bridges
        )
        return sidechains, backbones


# Helper functions for backbone/sidechain classification
def find_shortest_paths_between_stars(graph):
    star_nodes = [
        node for node, data in graph.nodes(data=True) if data["element"] == "*"
    ]
    shortest_paths = []
    for i in range(len(star_nodes)):
        for j in range(i + 1, len(star_nodes)):
            try:
                path = nx.shortest_path(
                    graph, source=star_nodes[i], target=star_nodes[j]
                )
                shortest_paths.append(path)
            except nx.NetworkXNoPath:
                continue
    return shortest_paths


def find_cycles_including_paths(graph, paths):
    all_cycles = nx.cycle_basis(graph)
    path_nodes = {node for path in paths for node in path}
    cycles_including_paths = [
        cycle for cycle in all_cycles if any(node in path_nodes for node in cycle)
    ]
    unique_cycles = {
        tuple(sorted((min(c), max(c))
              for c in zip(cycle, cycle[1:] + [cycle[0]])))
        for cycle in cycles_including_paths
    }
    return [list(cycle) for cycle in unique_cycles]


def add_degree_one_nodes_to_backbone(graph, backbone):
    for node in list(graph.nodes):
        if graph.degree[node] == 1:
            neighbor = next(iter(graph.neighbors(node)))
            if neighbor in backbone:
                backbone.append(node)
    return backbone


def classify_backbone_and_sidechains(graph):
    shortest_paths = find_shortest_paths_between_stars(graph)
    cycles = find_cycles_including_paths(graph, shortest_paths)
    backbone_nodes = set()
    for cycle in cycles:
        for edge in cycle:
            backbone_nodes.update(edge)
    for path in shortest_paths:
        backbone_nodes.update(path)
    backbone_nodes = add_degree_one_nodes_to_backbone(
        graph, list(backbone_nodes))
    sidechain_nodes = [
        node for node in graph.nodes if node not in backbone_nodes]
    return list(set(backbone_nodes)), sidechain_nodes


def get_real_backbone_and_sidechain_bridges(graph, backbone_nodes, sidechain_nodes):
    backbone_bridges = []
    sidechain_bridges = []
    for edge in graph.edges:
        start_node, end_node = edge
        if start_node in backbone_nodes and end_node in backbone_nodes:
            backbone_bridges.append(edge)
        elif start_node in sidechain_nodes or end_node in sidechain_nodes:
            sidechain_bridges.append(edge)
    return backbone_bridges, sidechain_bridges


def number_and_length_of_sidechains_and_backbones(sidechain_bridges, backbone_bridges):
    sidechain_graph = nx.Graph(sidechain_bridges)
    backbone_graph = nx.Graph(backbone_bridges)
    sidechains = list(nx.connected_components(sidechain_graph))
    backbones = list(nx.connected_components(backbone_graph))
    return sidechains, backbones

