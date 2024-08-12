from typing import List, Tuple, Optional
import networkx as nx
from rdkit import Chem


class Polymer:
    def __init__(self):
        self._psmiles: Optional[str] = None
        self._graph: Optional[nx.Graph] = None
        self._backbone_nodes: Optional[List[int]] = None
        self._sidechain_nodes: Optional[List[int]] = None
        self._connection_points: Optional[List[int]] = None

    @classmethod
    def from_psmiles(cls, psmiles: str) -> "Polymer":
        """
        Create a Polymer instance from a pSMILES string.

        Args:
            psmiles (str): The pSMILES representation of the polymer.

        Returns:
            Polymer: A new Polymer instance.

        Raises:
            ValueError: If the pSMILES string is invalid.
        """
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
            raise ValueError(f"Error processing pSMILES: {str(e)}")

    def _mol_to_nx(self, mol: Chem.Mol) -> nx.Graph:
        """Convert an RDKit molecule to a NetworkX graph."""
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
        """Identify connection points (asterisks) in the polymer graph."""
        self._connection_points = [
            node
            for node, data in self._graph.nodes(data=True)
            if data["element"] == "*"
        ]

    def _identify_backbone_and_sidechain(self):
        """Identify backbone and sidechain nodes in the polymer graph."""
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
        """
        Get RDKit molecules representing the backbone and sidechains.

        Returns:
            Tuple[List[Chem.Mol], List[Chem.Mol]]: Lists of backbone and sidechain molecules.
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
        """
        Get NetworkX graphs representing the backbone and sidechains.

        Returns:
            Tuple[nx.Graph, List[nx.Graph]]: The backbone graph and a list of sidechain graphs.
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
        """Convert a NetworkX subgraph back to an RDKit molecule."""
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
        """
        Calculate the molecular weight of the polymer.

        Returns:
            float: The molecular weight of the polymer.
        """
        mol = Chem.MolFromSmiles(self._psmiles)
        return Chem.Descriptors.ExactMolWt(mol)

    def get_connection_points(self) -> List[int]:
        """
        Get the connection points of the polymer.

        Returns:
            List[int]: The node indices of the connection points.
        """
        return self._connection_points


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
    cycles = set()
    for path in paths:
        for node in path:
            try:
                all_cycles = nx.cycle_basis(graph, node)
                for cycle in all_cycles:
                    if any(n in path for n in cycle):
                        sorted_cycle = tuple(
                            sorted(
                                (min(c), max(c))
                                for c in zip(cycle, cycle[1:] + [cycle[0]])
                            )
                        )
                        cycles.add(sorted_cycle)
            except nx.NetworkXNoCycle:
                continue
    return [list(cycle) for cycle in cycles]


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
    backbone_nodes = add_degree_one_nodes_to_backbone(graph, list(backbone_nodes))
    sidechain_nodes = [node for node in graph.nodes if node not in backbone_nodes]
    return list(set(backbone_nodes)), sidechain_nodes
