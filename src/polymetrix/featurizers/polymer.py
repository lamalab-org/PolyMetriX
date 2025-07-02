from typing import List, Optional, Tuple

import networkx as nx
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

import numpy as np
import torch
from torch.nn.functional import cosine_similarity


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
        self._bigsmiles: Optional[str] = None
        self._psmiles_embed: Optional[np.ndarray] = None
        self._bigsmiles_embed: Optional[np.ndarray] = None
        self._graph: Optional[nx.Graph] = None
        self._backbone_nodes: Optional[list[int]] = None
        self._sidechain_nodes: Optional[list[int]] = None
        self._connection_points: Optional[list[int]] = None

    @classmethod
    def from_psmiles(
        cls,
        psmiles: str,
        psmiles_embed: Optional[np.ndarray] = None,
        dataset=None,
        embed_column: str = "meta.psmiles_embed",
    ) -> "Polymer":
        """Creates a Polymer instance from a pSMILES string.

        Args:
            psmiles: str, the pSMILES string representing the polymer molecule.
            psmiles_embed: Optional[np.ndarray], the embedding for the pSMILES string.
            dataset: Optional dataset to lookup embedding if not provided.
            embed_column: str, the column name for pSMILES embeddings in the dataset.

        Returns:
            Polymer: A new Polymer object initialized with the given pSMILES string.

        Raises:
            ValueError: If the pSMILES string is invalid or not found in dataset.
        """
        polymer = cls()
        polymer.psmiles = psmiles
        if psmiles_embed is not None:
            polymer._psmiles_embed = psmiles_embed
        elif dataset is not None:
            polymer._psmiles_embed = polymer._lookup_embedding(
                dataset, psmiles, "psmiles", embed_column
            )
        return polymer

    @classmethod
    def from_bigsmiles(
        cls,
        bigsmiles: str,
        bigsmiles_embed: Optional[np.ndarray] = None,
        dataset=None,
        embed_column: str = "meta.bigsmiles_embed",
    ) -> "Polymer":
        """Creates a Polymer instance from a BIGSMILES string.

        Args:
            bigsmiles: str, the BIGSMILES string representing the polymer molecule.
            bigsmiles_embed: Optional[np.ndarray], the embedding for the BIGSMILES string.
            dataset: Optional dataset to lookup embedding if not provided.
            embed_column: str, the column name for BIGSMILES embeddings in the dataset.

        Returns:
            Polymer: A new Polymer object initialized with the given BIGSMILES string.

        Raises:
            ValueError: If the BIGSMILES string is not found in dataset.
        """
        polymer = cls()
        polymer._bigsmiles = bigsmiles
        if bigsmiles_embed is not None:
            polymer._bigsmiles_embed = bigsmiles_embed
        elif dataset is not None:
            polymer._bigsmiles_embed = polymer._lookup_embedding(
                dataset, bigsmiles, "bigsmiles", embed_column
            )
        return polymer

    def _lookup_embedding(
        self, dataset, string: str, string_type: str, embed_column: str
    ) -> np.ndarray:
        """Looks up an embedding in a dataset for a given string."""
        try:
            string_list = list(dataset.__getattribute__(string_type))
            idx = string_list.index(string)
            return dataset.get_meta([idx], [embed_column])[0]
        except ValueError as e:
            raise ValueError(
                f"{string_type.upper()} string not found in dataset: {string}"
            ) from e

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
            raise ValueError(f"Error processing pSMILES: {e!s}") from e

    @property
    def bigsmiles(self) -> Optional[str]:
        """Gets the BIGSMILES string of the polymer.

        Returns:
            Optional[str]: The BIGSMILES string, or None if not set.
        """
        return self._bigsmiles

    @bigsmiles.setter
    def bigsmiles(self, value: str):
        """Sets the BIGSMILES string.

        Args:
            value: str, the BIGSMILES string to set.
        """
        self._bigsmiles = value

    def set_psmiles_embed(self, embed: np.ndarray):
        """Sets the pSMILES embedding.

        Args:
            embed: np.ndarray, the embedding array for the pSMILES string.
        """
        self._psmiles_embed = embed

    def set_bigsmiles_embed(self, embed: np.ndarray):
        """Sets the BIGSMILES embedding.

        Args:
            embed: np.ndarray, the embedding array for the BIGSMILES string.
        """
        self._bigsmiles_embed = embed

    def _parse_embedding(self, embed: np.ndarray) -> torch.Tensor:
        """Converts an embedding (array or string) to a torch.Tensor.

        Args:

        embed: np.ndarray or str, the embedding to parse.
        """
        if isinstance(embed, np.ndarray) and embed.dtype == object:
            embed_data = np.fromstring(embed[0].strip("[]"), sep=" ")
        else:
            embed_data = (
                np.fromstring(embed.strip("[]"), sep=" ")
                if isinstance(embed, str)
                else embed
            )
        return torch.tensor(embed_data, dtype=torch.float32).unsqueeze(0)

    def polymer_dist(self, other: "Polymer") -> float:
        """Calculate cosine similarity between two polymers using their embeddings.

        Args:
            other: Polymer, another polymer to compare with.

        Returns:
            float: Cosine similarity between the two polymers.

        Raises:
            ValueError: If embeddings are not available or representation types don't match.
        """
        if self._psmiles_embed is not None and other._psmiles_embed is not None:
            embed1 = self._parse_embedding(self._psmiles_embed)
            embed2 = self._parse_embedding(other._psmiles_embed)
        elif self._bigsmiles_embed is not None and other._bigsmiles_embed is not None:
            embed1 = self._parse_embedding(self._bigsmiles_embed)
            embed2 = self._parse_embedding(other._bigsmiles_embed)
        else:
            raise ValueError(
                "Both polymers must have embeddings of the same type (pSMILES or BIGSMILES)"
            )
        return cosine_similarity(embed1, embed2, dim=1).item()

    def __eq__(self, other: "Polymer", threshold: float = 0.95) -> bool:
        """Check if two polymers are equal based on embedding similarity.

        Args:
            other: Polymer, another polymer to compare with.
            threshold: float, similarity threshold for considering polymers equal.

        Returns:
            bool: True if polymers are considered equal, False otherwise.
        """
        if not isinstance(other, Polymer):
            return False
        try:
            return self.polymer_dist(other) >= threshold
        except ValueError:
            return False

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
    def backbone_nodes(self) -> list[int]:
        """Gets the list of backbone node indices.

        Returns:
            list[int]: List of node indices representing the backbone.
        """
        return self._backbone_nodes

    @property
    def sidechain_nodes(self) -> list[int]:
        """Gets the list of sidechain node indices.

        Returns:
            list[int]: List of node indices representing the sidechains.
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
    ) -> Tuple[list[Chem.Mol], list[Chem.Mol]]:
        """Extracts RDKit molecule objects for the backbone and sidechains.

        Returns:
            Tuple[list[Chem.Mol], list[Chem.Mol]]: A tuple containing a list with the backbone
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

    def get_backbone_and_sidechain_graphs(self) -> tuple[nx.Graph, list[nx.Graph]]:
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

    def get_connection_points(self) -> list[int]:
        """Gets the list of connection point node indices.

        Returns:
            List[int]: List of node indices representing connection points.
        """
        return self._connection_points


# Helper functions for backbone/sidechain classification
def find_shortest_paths_between_stars(graph: nx.Graph) -> list[list[int]]:
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
    graph: nx.Graph, paths: list[list[int]]
) -> list[list[int]]:
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


def add_degree_one_nodes_to_backbone(graph: nx.Graph, backbone: List[int]) -> list[int]:
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


def classify_backbone_and_sidechains(graph: nx.Graph) -> tuple[list[int], list[int]]:
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
