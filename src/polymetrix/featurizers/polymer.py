from typing import List, Optional, Tuple, Dict

import networkx as nx
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt


class Polymer:
    """Represents a polymer molecule with its backbone and sidechain information.

    Attributes:
        psmiles: Optional[str], the pSMILES string of the polymer.
        terminal_groups: Optional[Dict[int, str]], maps node indices to terminal group SMILES.
        graph: Optional[nx.Graph], the NetworkX graph of the polymer structure.
        backbone_nodes: Optional[List[int]], node indices forming the backbone.
        sidechain_nodes: Optional[List[int]], node indices forming the sidechains.
        connection_points: Optional[List[int]], node indices of connection points.
        _mol: Optional[Chem.Mol], the RDKit molecule object (internal use).
    """

    def __init__(self):
        self._psmiles = None
        self.terminal_groups = None
        self.graph = None
        self.backbone_nodes = None
        self.sidechain_nodes = None
        self.connection_points = None
        self._mol = None

    @classmethod
    def from_psmiles(cls, psmiles: str) -> "Polymer":
        """Creates a Polymer instance from a pSMILES string.

        Args:
            psmiles: The pSMILES string representing the polymer.

        Returns:
            A new Polymer instance.

        Raises:
            ValueError: If the pSMILES string is invalid.
        """
        polymer = cls()
        polymer.psmiles = psmiles
        return polymer

    @property
    def psmiles(self) -> Optional[str]:
        """The pSMILES string of the polymer."""
        return self._psmiles

    @psmiles.setter
    def psmiles(self, value: str):
        """Sets the pSMILES string and updates the polymer's structure.

        Args:
            value: The pSMILES string to set.

        Raises:
            ValueError: If the pSMILES string is None, empty, or invalid.
        """
        if not value or not isinstance(value, str):
            raise ValueError("pSMILES cannot be None or empty")
        try:
            mol = Chem.MolFromSmiles(value)
            if mol is None:
                raise ValueError("Invalid pSMILES string")
            self._psmiles = value
            self._mol = mol
            self.graph = self._mol_to_nx(mol)
            self._identify_connection_points()
            self._identify_backbone_and_sidechain()
        except Exception as e:
            raise ValueError(f"Error processing pSMILES: {str(e)}") from e

    @property
    def terminal_groups(self) -> Optional[Dict[int, str]]:
        """Maps node indices to terminal group SMILES."""
        return self._terminal_groups

    @terminal_groups.setter
    def terminal_groups(self, value: Dict[int, str]):
        """Sets terminal groups for specific node positions.

        Args:
            value: Mapping of node indices to terminal group SMILES.
        """
        self._terminal_groups = value

    @staticmethod
    def _mol_to_nx(mol: Chem.Mol) -> nx.Graph:
        """Converts an RDKit molecule to a NetworkX graph.

        Args:
            mol: The RDKit molecule to convert.

        Returns:
            A NetworkX graph representing the molecule's structure.
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
        self.connection_points = [
            node for node, data in self.graph.nodes(data=True) if data["element"] == "*"
        ]

    def _identify_backbone_and_sidechain(self):
        """Classifies nodes into backbone and sidechain components."""
        self.backbone_nodes, self.sidechain_nodes = classify_backbone_and_sidechains(
            self.graph
        )

    def get_backbone_and_sidechain_molecules(
        self,
    ) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
        """Extracts RDKit molecules for the backbone and sidechains.

        Returns:
            A tuple of (list of backbone molecules, list of sidechain molecules).
        """
        if self.terminal_groups:
            backbone_mol = self._create_backbone_with_terminal_groups()
        else:
            backbone_mol = self._subgraph_to_mol(
                self.graph.subgraph(self.backbone_nodes)
            )

        sidechain_mols = [
            self._subgraph_to_mol(self.graph.subgraph(nodes))
            for nodes in nx.connected_components(
                self.graph.subgraph(self.sidechain_nodes)
            )
        ]
        return [backbone_mol], sidechain_mols

    def _create_backbone_with_terminal_groups(self) -> Chem.Mol:
        """Creates a backbone molecule with terminal groups applied.

        Returns:
            The RDKit molecule for the backbone with terminal groups.
        """
        backbone_subgraph = self.graph.subgraph(self.backbone_nodes)
        mol = Chem.RWMol()
        node_to_idx = {}

        for node in backbone_subgraph.nodes():
            if node in self.terminal_groups:
                terminal_smiles = self.terminal_groups[node]
                terminal_mol = Chem.MolFromSmiles(terminal_smiles)
                if terminal_mol:
                    atom = terminal_mol.GetAtomWithIdx(0)
                    new_atom = Chem.Atom(atom.GetAtomicNum())
                    new_atom.SetFormalCharge(atom.GetFormalCharge())
                    idx = mol.AddAtom(new_atom)
                    node_to_idx[node] = idx
                else:
                    print(
                        f"Warning: Invalid terminal group SMILES '{terminal_smiles}' for node {node}"
                    )
                    atom = Chem.Atom(6)  # Carbon
                    idx = mol.AddAtom(atom)
                    node_to_idx[node] = idx
            else:
                atom = Chem.Atom(backbone_subgraph.nodes[node]["atomic_num"])
                atom.SetFormalCharge(
                    backbone_subgraph.nodes[node].get("formal_charge", 0)
                )
                idx = mol.AddAtom(atom)
                node_to_idx[node] = idx

        for u, v, data in backbone_subgraph.edges(data=True):
            mol.AddBond(node_to_idx[u], node_to_idx[v], data["bond_type"])

        result = mol.GetMol()
        return result if result else self._subgraph_to_mol(backbone_subgraph)

    def get_backbone_and_sidechain_graphs(self) -> Tuple[nx.Graph, List[nx.Graph]]:
        """Extracts NetworkX graphs for the backbone and sidechains.

        Returns:
            A tuple of (backbone graph, list of sidechain graphs).
        """
        backbone_graph = self.graph.subgraph(self.backbone_nodes)
        sidechain_graphs = [
            self.graph.subgraph(nodes)
            for nodes in nx.connected_components(
                self.graph.subgraph(self.sidechain_nodes)
            )
        ]
        return backbone_graph, sidechain_graphs

    @staticmethod
    def _subgraph_to_mol(subgraph: nx.Graph) -> Chem.Mol:
        """Converts a NetworkX subgraph to an RDKit molecule.

        Args:
            subgraph: The subgraph to convert.

        Returns:
            The RDKit molecule created from the subgraph.
        """
        mol = Chem.RWMol()
        node_to_idx = {}
        for node in subgraph.nodes():
            atom = Chem.Atom(subgraph.nodes[node]["atomic_num"])
            atom.SetFormalCharge(subgraph.nodes[node].get("formal_charge", 0))
            idx = mol.AddAtom(atom)
            node_to_idx[node] = idx
        for u, v, data in subgraph.edges(data=True):
            mol.AddBond(node_to_idx[u], node_to_idx[v], data["bond_type"])
        return mol.GetMol()

    def calculate_molecular_weight(self) -> float:
        """Calculates the exact molecular weight of the polymer.

        Returns:
            The molecular weight of the polymer.
        """
        return ExactMolWt(self._mol) if self._mol else 0.0

    def get_connection_points(self) -> List[int]:
        """Gets the connection point node indices.

        Returns:
            List of node indices representing connection points.
        """
        return self.connection_points


# Helper functions for backbone/sidechain classification
def find_shortest_paths_between_stars(graph: nx.Graph) -> List[List[int]]:
    """Finds shortest paths between all pairs of asterisk (*) nodes in the graph.

    Args:
        graph: The input graph to analyze.

    Returns:
        List of shortest paths, where each path is a list of node indices.
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
    """Identifies cycles that include nodes from the given paths.

    Args:
        graph: The input graph to analyze.
        paths: List of paths whose nodes are used to filter cycles.

    Returns:
        List of cycles, where each cycle is a list of node indices.
    """
    all_cycles = nx.cycle_basis(graph)
    path_nodes = {node for path in paths for node in path}
    return [cycle for cycle in all_cycles if any(node in path_nodes for node in cycle)]


def add_degree_one_nodes_to_backbone(graph: nx.Graph, backbone: List[int]) -> List[int]:
    """Adds degree-1 nodes connected to backbone nodes to the backbone list, avoiding duplicates.

    Args:
        graph: The input graph to analyze.
        backbone: Initial list of backbone node indices.

    Returns:
        Updated backbone list including degree-1 nodes, with no duplicates.
    """
    for node in graph.nodes:
        if graph.degree[node] == 1 and node not in backbone:
            neighbor = next(iter(graph.neighbors(node)))
            if neighbor in backbone:
                backbone.append(node)
    return backbone


def classify_backbone_and_sidechains(graph: nx.Graph) -> Tuple[List[int], List[int]]:
    """Classifies nodes into backbone and sidechain components based on paths and cycles.

    Args:
        graph: The input graph to classify.

    Returns:
        A tuple of (backbone nodes, sidechain nodes).
    """
    shortest_paths = find_shortest_paths_between_stars(graph)
    cycles = find_cycles_including_paths(graph, shortest_paths)
    backbone_nodes = set()
    for cycle in cycles:
        backbone_nodes.update(cycle)
    for path in shortest_paths:
        backbone_nodes.update(path)
    backbone_nodes = add_degree_one_nodes_to_backbone(graph, list(backbone_nodes))
    sidechain_nodes = [node for node in graph.nodes if node not in backbone_nodes]
    return list(backbone_nodes), sidechain_nodes
