from typing import List, Optional, Tuple

import networkx as nx
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt


class Polymer:
    """A class to represent a polymer molecule and extract its backbone and sidechain information.

    Attributes:
        psmiles: Optional[str], the pSMILES string representing the polymer molecule.
        graph: Optional[nx.Graph], a NetworkX graph representing the polymer structure.
        backbone_nodes: Optional[List[int]], list of node indices forming the polymer backbone.
        sidechain_nodes: Optional[List[int]], list of node indices forming the sidechains.
        connection_points: Optional[List[int]], list of node indices representing connection points.

    Raises:
        ValueError: If the provided pSMILES string is invalid or cannot be processed.
    """

    def __init__(self):
        self._psmiles: Optional[str] = None
        self._graph: Optional[nx.Graph] = None
        self._backbone_nodes: Optional[List[int]] = None
        self._sidechain_nodes: Optional[List[int]] = None
        self._connection_points: Optional[List[int]] = None

    @classmethod
    def from_psmiles(cls, psmiles: str) -> "Polymer":
        """Creates a Polymer instance from a pSMILES string.

        Args:
            psmiles: str, the pSMILES string representing the polymer molecule.

        Returns:
            Polymer: A new Polymer object initialized with the given pSMILES string.

        Raises:
            ValueError: If the pSMILES string is invalid.
        """
        polymer = cls()
        polymer.psmiles = psmiles
        return polymer

    @property
    def psmiles(self) -> Optional[str]:
        """Gets the pSMILES string of the polymer.

        Returns:
            Optional[str]: The pSMILES string, or None if not set.
        """
        return self._psmiles

    @psmiles.setter
    def psmiles(self, value: str):
        """Sets the pSMILES string and updates the polymer's internal structure.

        Args:
            value: str, the pSMILES string to set.

        Raises:
            ValueError: If the pSMILES string is invalid or cannot be processed.
        """
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
        """Converts an RDKit molecule to a NetworkX graph.

        Args:
            mol: Chem.Mol, the RDKit molecule object to convert.

        Returns:
            nx.Graph: A NetworkX graph representing the molecule's structure.
        """
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(
                atom.GetIdx(),
                atomic_num=atom.GetAtomicNum(),
                element=atom.GetSymbol(),
                formal_charge=atom.GetFormalCharge(),
                is_aromatic=atom.GetIsAromatic(),
            )
        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond_type=bond.GetBondType(),
                is_aromatic=bond.GetIsAromatic(),
            )
        return G

    def _identify_connection_points(self):
        """Identifies connection points (asterisk atoms) in the polymer graph."""
        self._connection_points = [
            node
            for node, data in self._graph.nodes(data=True)
            if data["element"] == "*"
        ]

    def _identify_backbone_and_sidechain(self):
        """Classifies nodes into backbone and sidechain components."""
        self._backbone_nodes, self._sidechain_nodes = classify_backbone_and_sidechains(
            self._graph
        )

    @property
    def backbone_nodes(self) -> List[int]:
        """Gets the list of backbone node indices.

        Returns:
            List[int]: List of node indices representing the backbone.
        """
        return self._backbone_nodes

    @property
    def sidechain_nodes(self) -> List[int]:
        """Gets the list of sidechain node indices.

        Returns:
            List[int]: List of node indices representing the sidechains.
        """
        return self._sidechain_nodes

    @property
    def graph(self) -> nx.Graph:
        """Gets the NetworkX graph of the polymer.

        Returns:
            nx.Graph: The graph representing the polymer structure.
        """
        return self._graph

    def get_backbone_and_sidechain_molecules(
        self,
    ) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
        """Extracts RDKit molecule objects for the backbone and sidechains.

        Returns:
            Tuple[List[Chem.Mol], List[Chem.Mol]]: A tuple containing a list with the backbone
                molecule and a list of sidechain molecules.
        """
        backbone_mol = self._subgraph_to_mol(self._graph.subgraph(self._backbone_nodes))
        sidechain_mols = [
            self._subgraph_to_mol(self._graph.subgraph(nodes))
            for nodes in nx.connected_components(
                self._graph.subgraph(self._sidechain_nodes)
            )
        ]
        return [backbone_mol], sidechain_mols

    def get_backbone_and_sidechain_graphs(self) -> Tuple[nx.Graph, List[nx.Graph]]:
        """Extracts NetworkX graphs for the backbone and sidechains.

        Returns:
            Tuple[nx.Graph, List[nx.Graph]]: A tuple containing the backbone graph and a list
                of sidechain graphs.
        """
        backbone_graph = self._graph.subgraph(self._backbone_nodes)
        sidechain_graphs = [
            self._graph.subgraph(nodes)
            for nodes in nx.connected_components(
                self._graph.subgraph(self._sidechain_nodes)
            )
        ]
        return [backbone_graph], sidechain_graphs

    def _subgraph_to_mol(self, subgraph: nx.Graph) -> Chem.Mol:
        """Converts a NetworkX subgraph to an RDKit molecule.

        Args:
            subgraph: nx.Graph, the subgraph to convert.

        Returns:
            Chem.Mol: The RDKit molecule object created from the subgraph.
        """
        mol = Chem.RWMol()
        node_to_idx = {}
        for node in subgraph.nodes():
            atom = Chem.Atom(subgraph.nodes[node]["atomic_num"])
            if "formal_charge" in subgraph.nodes[node]:
                atom.SetFormalCharge(subgraph.nodes[node]["formal_charge"])
            idx = mol.AddAtom(atom)
            node_to_idx[node] = idx
        for u, v, data in subgraph.edges(data=True):
            mol.AddBond(node_to_idx[u], node_to_idx[v], data["bond_type"])
        return mol.GetMol()

    def calculate_molecular_weight(self) -> float:
        """Calculates the exact molecular weight of the polymer.

        Returns:
            float: The molecular weight of the polymer molecule.
        """
        mol = Chem.MolFromSmiles(self._psmiles)
        return ExactMolWt(mol)

    def get_connection_points(self) -> List[int]:
        """Gets the list of connection point node indices.

        Returns:
            List[int]: List of node indices representing connection points.
        """
        return self._connection_points


# Helper functions for backbone/sidechain classification
def find_shortest_paths_between_stars(graph: nx.Graph) -> List[List[int]]:
    """Finds shortest paths between all pairs of asterisk (*) nodes in the graph.

    Args:
        graph: nx.Graph, the input graph to analyze.

    Returns:
        List[List[int]]: A list of shortest paths, where each path is a list of node indices.
    """
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


def find_cycles_including_paths(
    graph: nx.Graph, paths: List[List[int]]
) -> List[List[int]]:
    """Identifies cycles in the graph that include nodes from the given paths.

    Args:
        graph: nx.Graph, the input graph to analyze.
        paths: List[List[int]], list of paths whose nodes are used to filter cycles.

    Returns:
        List[List[int]]: A list of unique cycles, where each cycle is a list of node indices.
    """
    all_cycles = nx.cycle_basis(graph)
    path_nodes = {node for path in paths for node in path}
    cycles_including_paths = [
        cycle for cycle in all_cycles if any(node in path_nodes for node in cycle)
    ]
    unique_cycles = {
        tuple(sorted((min(c), max(c)) for c in zip(cycle, cycle[1:] + [cycle[0]])))
        for cycle in cycles_including_paths
    }
    return [list(cycle) for cycle in unique_cycles]


def add_degree_one_nodes_to_backbone(graph: nx.Graph, backbone: List[int]) -> List[int]:
    """Adds degree-1 nodes connected to backbone nodes to the backbone list.

    Args:
        graph: nx.Graph, the input graph to analyze.
        backbone: List[int], the initial list of backbone node indices.

    Returns:
        List[int]: The updated backbone list including degree-1 nodes.
    """
    for node in list(graph.nodes):
        if graph.degree[node] == 1:
            neighbor = next(iter(graph.neighbors(node)))
            if neighbor in backbone:
                backbone.append(node)
    return backbone


def classify_backbone_and_sidechains(graph: nx.Graph) -> Tuple[List[int], List[int]]:
    """Classifies nodes into backbone and sidechain components based on paths and cycles.

    Args:
        graph: nx.Graph, the input graph to classify.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing the list of backbone nodes and
            the list of sidechain nodes.
    """
    shortest_paths = find_shortest_paths_between_stars(graph)
    cycles = find_cycles_including_paths(graph, shortest_paths)
    backbone_nodes = set()
    for cycle in cycles:
        for edge in cycle:
            backbone_nodes.update(edge)
    for path in shortest_paths:
        backbone_nodes.update(path)
    backbone_nodes = add_degree_one_nodes_to_backbone(graph, list(backbone_nodes))
    sidechain_nodes = [node for node in graph.nodes if node not in backbone_nodes]
    return list(set(backbone_nodes)), sidechain_nodes
