from typing import List, Optional, Tuple, Dict
import networkx as nx
from rdkit import Chem
from rdkit.Chem import RWMol, Atom, Bond
from rdkit.Chem.Descriptors import ExactMolWt
import logging

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Polymer:
    """Represents a polymer molecule with its backbone and sidechain information.

    Attributes:
        psmiles: Optional[str], the pSMILES string of the polymer.
        backbone_terminal_groups: Optional[Dict[str, str]], maps connection point patterns to backbone terminal group SMILES.
        sidechain_terminal_groups: Optional[Dict[str, str]], maps connection point patterns to sidechain terminal group SMILES.
        graph: Optional[nx.Graph], the NetworkX graph of the polymer structure.
        backbone_nodes: Optional[List[int]], node indices forming the backbone.
        sidechain_nodes: Optional[List[int]], node indices forming the sidechains.
        connection_points: Optional[List[int]], node indices of connection points.
        _mol: Optional[Chem.Mol], the RDKit molecule object (internal use).
    """

    def __init__(self):
        self._psmiles = None
        self._backbone_terminal_groups = None
        self._sidechain_terminal_groups = None
        self.graph = None
        self.backbone_nodes = None
        self.sidechain_nodes = None
        self.connection_points = None
        self._mol = None

    @property
    def mol(self) -> Optional[Chem.Mol]:
        """Returns the full polymer molecule, compatible with featurizers expecting a 'mol' attribute."""
        return self.full_polymer_mol

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
    def backbone_terminal_groups(self) -> Optional[Dict[str, str]]:
        """Maps connection point patterns to backbone terminal group SMILES."""
        return self._backbone_terminal_groups

    @backbone_terminal_groups.setter
    def backbone_terminal_groups(self, value: Dict[str, str]):
        """Sets terminal groups for backbone connection points."""
        self._backbone_terminal_groups = value

    @property
    def sidechain_terminal_groups(self) -> Optional[Dict[str, str]]:
        """Maps connection point patterns to sidechain terminal group SMILES."""
        return self._sidechain_terminal_groups

    @sidechain_terminal_groups.setter
    def sidechain_terminal_groups(self, value: Dict[str, str]):
        """Sets terminal groups for sidechain connection points."""
        self._sidechain_terminal_groups = value

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

    @property
    def backbone_molecule(self) -> Chem.Mol:
        """Gets the backbone molecule."""
        return self._get_backbone_molecule(include_terminal_groups=True)

    def _get_backbone_molecule(self, include_terminal_groups: bool = True) -> Chem.Mol:
        """Internal method to get backbone molecule with optional terminal groups."""
        backbone_mol = self._extract_substructure_mol(self.backbone_nodes)
        if include_terminal_groups and self._backbone_terminal_groups:
            backbone_mol = insert_terminal_group(
                backbone_mol, self._backbone_terminal_groups, is_sidechain=False
            )
        return backbone_mol

    @property
    def sidechain_molecules(self) -> List[Chem.Mol]:
        """Gets the sidechain molecules."""
        return self._get_sidechain_molecules(include_terminal_groups=True)

    def _get_sidechain_molecules(
        self, include_terminal_groups: bool = True
    ) -> List[Chem.Mol]:
        """Internal method to get sidechain molecules with optional terminal groups."""
        sidechain_components = list(
            nx.connected_components(self.graph.subgraph(self.sidechain_nodes))
        )
        sidechain_mols = []
        for component_nodes in sidechain_components:
            mol = self._extract_substructure_mol(list(component_nodes))
            if include_terminal_groups and self._sidechain_terminal_groups:
                mol = insert_terminal_group(
                    mol, self._sidechain_terminal_groups, is_sidechain=True
                )
            sidechain_mols.append(mol)
        return sidechain_mols

    @property
    def full_polymer_mol(self) -> Chem.Mol:
        """Gets the full polymer molecule."""
        return self._get_full_polymer_mol(include_terminal_groups=True)

    def _get_full_polymer_mol(self, include_terminal_groups: bool = True) -> Chem.Mol:
        """Internal method to get full polymer molecule with optional terminal groups."""
        if include_terminal_groups and self._backbone_terminal_groups:
            return insert_terminal_group(
                self._mol, self._backbone_terminal_groups, is_sidechain=False
            )
        return self._mol

    def _extract_substructure_mol(self, node_indices: List[int]) -> Chem.Mol:
        """Extracts a substructure molecule from the main molecule using node indices."""
        if not node_indices:
            return Chem.MolFromSmiles("")
        mol = RWMol()
        old_to_new_idx = {}
        for old_idx in node_indices:
            atom = self._mol.GetAtomWithIdx(old_idx)
            new_atom = Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            if atom.GetIsAromatic():
                new_atom.SetIsAromatic(True)
            new_idx = mol.AddAtom(new_atom)
            old_to_new_idx[old_idx] = new_idx
        for bond in self._mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if begin_idx in old_to_new_idx and end_idx in old_to_new_idx:
                mol.AddBond(
                    old_to_new_idx[begin_idx],
                    old_to_new_idx[end_idx],
                    bond.GetBondType(),
                )
        return mol.GetMol()

    def get_backbone_and_sidechain_molecules(
        self,
    ) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
        """Extracts RDKit molecules for the backbone and sidechains.

        Returns:
            A tuple of (list of backbone molecules, list of sidechain molecules).
        """
        return [self.backbone_molecule], self.sidechain_molecules

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


def insert_terminal_group(
    mol: Chem.Mol, terminal_groups: Dict[str, str], is_sidechain: bool = False
) -> Chem.Mol:
    """Inserts terminal groups into a molecule by replacing connection points or attaching to sidechains.

    Args:
        mol: The RDKit molecule to modify.
        terminal_groups: Dictionary mapping patterns to terminal group SMILES.
        is_sidechain: If True, attach terminal groups to sidechains; else, replace backbone connection points.

    Returns:
        A new RDKit molecule with terminal groups inserted.
    """
    if not terminal_groups:
        return mol

    mol_copy = RWMol(mol)

    if is_sidechain:
        for pattern, terminal_smiles in terminal_groups.items():
            terminal_mol = Chem.MolFromSmiles(
                terminal_smiles.replace("*", "")
            )  # Remove asterisk for attachment
            if terminal_mol is None:
                logging.warning(f"Invalid terminal group SMILES '{terminal_smiles}'")
                continue
            target_idx = 0  # Attach to the first atom of the sidechain
            mol_copy = attach_terminal_to_atom(
                mol_copy, target_idx, terminal_mol, attachment_idx=None
            )
    else:
        asterisk_atoms = [
            atom.GetIdx() for atom in mol_copy.GetAtoms() if atom.GetSymbol() == "*"
        ]
        for pattern, terminal_smiles in terminal_groups.items():
            if pattern == "[*]":
                terminal_mol = Chem.MolFromSmiles(terminal_smiles)
                if terminal_mol is None:
                    logging.warning(
                        f"Invalid terminal group SMILES '{terminal_smiles}'"
                    )
                    continue
                attachment_idx = None
                for atom in terminal_mol.GetAtoms():
                    if atom.GetSymbol() == "*":
                        attachment_idx = atom.GetIdx()
                        break
                if attachment_idx is None:
                    logging.warning(
                        f"No attachment point (*) found in terminal group '{terminal_smiles}'"
                    )
                    continue
                for ast_idx in sorted(asterisk_atoms, reverse=True):
                    mol_copy = replace_asterisk_with_terminal(
                        mol_copy, ast_idx, terminal_mol, attachment_idx
                    )

    return mol_copy.GetMol()


def replace_asterisk_with_terminal(
    mol: RWMol, asterisk_idx: int, terminal_mol: Chem.Mol, attachment_idx: int
) -> RWMol:
    """Replaces a single asterisk atom with a terminal group.

    Args:
        mol: The molecule being modified.
        asterisk_idx: Index of the asterisk atom to replace.
        terminal_mol: The terminal group molecule.
        attachment_idx: Index of the attachment point in the terminal group.

    Returns:
        The modified molecule.
    """
    asterisk_atom = mol.GetAtomWithIdx(asterisk_idx)
    neighbors = [n.GetIdx() for n in asterisk_atom.GetNeighbors()]
    if not neighbors:
        for atom in terminal_mol.GetAtoms():
            if atom.GetSymbol() != "*":
                new_atom = Atom(atom.GetAtomicNum())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                mol.ReplaceAtom(asterisk_idx, new_atom)
                break
        return mol
    neighbor_idx = neighbors[0]
    bond = mol.GetBondBetweenAtoms(asterisk_idx, neighbor_idx)
    bond_type = bond.GetBondType() if bond else Chem.BondType.SINGLE
    mol.RemoveAtom(asterisk_idx)
    if neighbor_idx > asterisk_idx:
        neighbor_idx -= 1
    atom_mapping = {}
    for atom in terminal_mol.GetAtoms():
        if atom.GetIdx() != attachment_idx:
            new_atom = Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_idx = mol.AddAtom(new_atom)
            atom_mapping[atom.GetIdx()] = new_idx
    for bond in terminal_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx == attachment_idx or end_idx == attachment_idx:
            continue
        if begin_idx in atom_mapping and end_idx in atom_mapping:
            mol.AddBond(
                atom_mapping[begin_idx], atom_mapping[end_idx], bond.GetBondType()
            )
    for bond in terminal_mol.GetBonds():
        if bond.GetBeginAtomIdx() == attachment_idx:
            connection_atom_idx = bond.GetEndAtomIdx()
        elif bond.GetEndAtomIdx() == attachment_idx:
            connection_atom_idx = bond.GetBeginAtomIdx()
        else:
            continue
        if connection_atom_idx in atom_mapping:
            mol.AddBond(neighbor_idx, atom_mapping[connection_atom_idx], bond_type)
            break
    return mol


def attach_terminal_to_atom(
    mol: RWMol,
    target_idx: int,
    terminal_mol: Chem.Mol,
    attachment_idx: int = None,
) -> RWMol:
    """Attaches a terminal group to a specific atom in the molecule.

    Args:
        mol: The molecule being modified.
        target_idx: Index of the target atom to attach the terminal group.
        terminal_mol: The terminal group molecule.
        attachment_idx: Index of the attachment point in the terminal group (optional for sidechains).

    Returns:
        The modified molecule.
    """
    atom_mapping = {}
    for atom in terminal_mol.GetAtoms():
        new_atom = Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_idx = mol.AddAtom(new_atom)
        atom_mapping[atom.GetIdx()] = new_idx
    for bond in terminal_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in atom_mapping and end_idx in atom_mapping:
            mol.AddBond(
                atom_mapping[begin_idx], atom_mapping[end_idx], bond.GetBondType()
            )
    first_terminal_atom_idx = next(iter(atom_mapping.keys()))
    mol.AddBond(target_idx, atom_mapping[first_terminal_atom_idx], Chem.BondType.SINGLE)
    return mol


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
