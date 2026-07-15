from typing import List, Optional, Tuple, Dict, Any
import json

import networkx as nx
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from polymetrix.featurizers.polymer import classify_backbone_and_sidechains
from polymetrix.bigsmiles.bigsmiles_utils import (
    parse_bigsmiles,
    repeat_unit_graphs,
    normalize_bigsmiles,
    _iter_stochastic_objects,
    end_group_atoms,
    graph_to_mol,
    mol_to_smiles,
)


class RepeatUnit:
    """One stochastic fragment (repeat unit) of a BigSMILES polymer.

    Attributes:
        index: Position of this repeat unit within the polymer.
        graph: NetworkX graph of the repeat unit (bonding descriptors are ``*``).
        backbone_nodes: Node keys classified as backbone.
        sidechain_nodes: Node keys classified as sidechain.
        connection_points: Node keys of the bonding descriptors.
    """

    def __init__(self, index: int, graph: nx.Graph):
        self.index = index
        self.graph = graph
        backbone, sidechain = classify_backbone_and_sidechains(graph)
        self.backbone_nodes = backbone
        self.sidechain_nodes = sidechain
        self.connection_points = [
            node for node, data in graph.nodes(data=True) if data.get("element") == "*"
        ]

    @property
    def backbone_molecule(self) -> Chem.Mol:
        """RDKit mol of this repeat unit's backbone (``*``-terminated)."""
        return graph_to_mol(self.graph, self.backbone_nodes)

    @property
    def sidechain_molecules(self) -> List[Chem.Mol]:
        """RDKit mols of this repeat unit's sidechains (one per component)."""
        return [
            graph_to_mol(self.graph, list(component))
            for component in nx.connected_components(
                self.graph.subgraph(self.sidechain_nodes)
            )
        ]

    @property
    def backbone_smiles(self) -> str:
        """SMILES of the backbone."""
        return mol_to_smiles(self.backbone_molecule)

    @property
    def sidechain_smiles(self) -> List[str]:
        """SMILES of each sidechain."""
        return [mol_to_smiles(m) for m in self.sidechain_molecules]

    @property
    def full_molecule(self) -> Chem.Mol:
        """RDKit mol of the whole repeat unit (backbone + sidechains)."""
        return graph_to_mol(self.graph, list(self.graph.nodes))

    @property
    def backbone_and_sidechain_smiles(self) -> str:
        """Canonical pSMILES of the whole repeat unit (``*``-terminated)."""
        return mol_to_smiles(self.full_molecule)

    @property
    def descriptors(self) -> List[str]:
        """Bonding-descriptor labels ([$]/[<]/[>]) of the connection points."""
        return [
            self.graph.nodes[node].get("descriptor") for node in self.connection_points
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this repeat unit's classification to a plain dict."""
        return {
            "index": self.index,
            "backbone_smiles": self.backbone_smiles,
            "sidechain_smiles": self.sidechain_smiles,
            "descriptors": self.descriptors,
            "num_backbone_atoms": sum(
                1
                for n in self.backbone_nodes
                if self.graph.nodes[n].get("element") != "*"
            ),
            "num_sidechains": len(self.sidechain_molecules),
        }


class BigSmilesPolymer:
    """Represents a polymer parsed from a BigSMILES string.

    Attributes:
        bigsmiles: The BigSMILES string of the polymer.
        repeat_units: List of :class:`RepeatUnit`, one per stochastic fragment.
        end_groups: Element symbols of explicit end-group atoms (may be empty).
        graph: A combined NetworkX graph over all repeat units (namespaced keys).
        backbone_nodes / sidechain_nodes / connection_points: combined node keys.
    """

    def __init__(self):
        self._bigsmiles: Optional[str] = None
        self._bs_obj = None
        self.repeat_units: List[RepeatUnit] = []
        self.end_groups: List[str] = []
        self.graph: Optional[nx.Graph] = None
        self.backbone_nodes: Optional[List] = None
        self.sidechain_nodes: Optional[List] = None
        self.connection_points: Optional[List] = None

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_bigsmiles(cls, bigsmiles_str: str) -> "BigSmilesPolymer":
        """Create a ``BigSmilesPolymer`` from a BigSMILES string.

        Args:
            bigsmiles_str: The BigSMILES string.

        Returns:
            A new ``BigSmilesPolymer`` instance.

        Raises:
            ValueError: If the BigSMILES string is invalid or contains no
                stochastic repeat unit.
        """
        polymer = cls()
        polymer.bigsmiles = bigsmiles_str
        return polymer

    @property
    def bigsmiles(self) -> Optional[str]:
        """The BigSMILES string of the polymer."""
        return self._bigsmiles

    @bigsmiles.setter
    def bigsmiles(self, value: str):
        if not value or not isinstance(value, str):
            raise ValueError("BigSMILES cannot be None or empty")
        fragment_graphs = repeat_unit_graphs(value)
        if not fragment_graphs:
            raise ValueError(
                "No stochastic repeat unit found in BigSMILES "
                f"'{value}'. A repeat unit must be enclosed in {{...}}."
            )
        # End groups are read from the whole-string parse. For a single
        # stochastic object this is exact; block copolymers (concatenated
        # objects) can consume a middle block's descriptors at the junctions
        # and fail the whole-string parse — in that case fall back to no
        # recorded end groups (the repeat units are still parsed per block).
        try:
            self._bs_obj = parse_bigsmiles(value)
            self.end_groups = end_group_atoms(self._bs_obj)
        except ValueError:
            self._bs_obj = None
            self.end_groups = []
        self._bigsmiles = value
        self.repeat_units = [
            RepeatUnit(i, graph) for i, (frag, graph) in enumerate(fragment_graphs)
        ]
        self._build_combined_graph()

    def _build_combined_graph(self):
        """Build one graph spanning all repeat units with namespaced node keys.

        Keys become ``(unit_index, original_key)`` so nodes from different
        repeat units never collide; classification results are copied over.
        """
        combined = nx.Graph()
        backbone, sidechain, connections = [], [], []
        for unit in self.repeat_units:
            remap = {node: (unit.index, node) for node in unit.graph.nodes}
            for node, data in unit.graph.nodes(data=True):
                combined.add_node(remap[node], **data)
            for u, v, data in unit.graph.edges(data=True):
                combined.add_edge(remap[u], remap[v], **data)
            backbone.extend(remap[n] for n in unit.backbone_nodes)
            sidechain.extend(remap[n] for n in unit.sidechain_nodes)
            connections.extend(remap[n] for n in unit.connection_points)
        self.graph = combined
        self.backbone_nodes = backbone
        self.sidechain_nodes = sidechain
        self.connection_points = connections

    # ------------------------------------------------------------------ #
    # Featurizer-facing interface (mirrors Polymer)
    # ------------------------------------------------------------------ #
    @property
    def mol(self) -> Chem.Mol:
        """Full polymer molecule (all repeat units), for generic featurizers."""
        return self.full_polymer_mol

    @property
    def full_polymer_mol(self) -> Chem.Mol:
        """Combined RDKit mol over every repeat unit (``*``-terminated)."""
        return graph_to_mol(self.graph, list(self.graph.nodes))

    @property
    def psmiles(self) -> str:
        """Equivalent pSMILES string (``*``-terminated repeat unit).

        This is the BigSMILES → pSMILES bridge. For a homopolymer it is the
        canonical pSMILES of the single repeat unit; for a copolymer the repeat
        units are returned as a dot-disconnected pSMILES. It lets pSMILES-only
        featurizers (e.g. ``FullPolymerFeaturizer``, which reads
        ``polymer.psmiles``) run on a BigSMILES polymer unchanged.
        """
        parts = [unit.backbone_and_sidechain_smiles for unit in self.repeat_units]
        return ".".join(p for p in parts if p)

    @property
    def backbone_molecule(self) -> Chem.Mol:
        """Backbone mol of the first repeat unit (parity with ``Polymer``)."""
        return self.repeat_units[0].backbone_molecule

    @property
    def backbone_molecules(self) -> List[Chem.Mol]:
        """Backbone mol of every repeat unit."""
        return [unit.backbone_molecule for unit in self.repeat_units]

    @property
    def sidechain_molecules(self) -> List[Chem.Mol]:
        """All sidechain mols across every repeat unit."""
        mols: List[Chem.Mol] = []
        for unit in self.repeat_units:
            mols.extend(unit.sidechain_molecules)
        return mols

    def get_backbone_and_sidechain_molecules(
        self,
    ) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
        """Return ``([backbone mols], [sidechain mols])``.

        Matches the tuple shape used by ``Polymer`` so the existing featurizers
        work unchanged. For a copolymer the backbone list has one entry per
        repeat unit and the sidechain list pools all sidechains.
        """
        return self.backbone_molecules, self.sidechain_molecules

    def get_backbone_and_sidechain_graphs(
        self,
    ) -> Tuple[List[nx.Graph], List[nx.Graph]]:
        """Return ``([backbone graphs], [sidechain graphs])`` across all units."""
        backbone_graphs = [
            unit.graph.subgraph(unit.backbone_nodes) for unit in self.repeat_units
        ]
        sidechain_graphs = []
        for unit in self.repeat_units:
            for component in nx.connected_components(
                unit.graph.subgraph(unit.sidechain_nodes)
            ):
                sidechain_graphs.append(unit.graph.subgraph(component))
        return backbone_graphs, sidechain_graphs

    def calculate_molecular_weight(self) -> float:
        """Exact molecular weight of the full (``*``-terminated) polymer mol."""
        mol = self.full_polymer_mol
        if mol is None or mol.GetNumAtoms() == 0:
            return 0.0
        try:
            work = Chem.Mol(mol)
            Chem.SanitizeMol(work)
            return ExactMolWt(work)
        except Exception:
            return ExactMolWt(mol)

    def get_connection_points(self) -> List:
        """Combined connection-point node keys."""
        return self.connection_points

    @property
    def num_repeat_units(self) -> int:
        """Number of stochastic fragments (repeat units)."""
        return len(self.repeat_units)

    @property
    def is_copolymer(self) -> bool:
        """True if the polymer has more than one distinct repeat unit."""
        return len(self.repeat_units) > 1

    @property
    def copolymer_type(self) -> str:
        """Classify the copolymer architecture from the BigSMILES notation.

        Following the BigSMILES paper (Lin et al., *ACS Cent. Sci.* 2019), the
        two ways of writing multi-component polymers map to distinct
        architectures:

        * ``"homopolymer"`` — a single repeat unit.
        * ``"random"`` — one stochastic object with comma-separated fragments,
          e.g. ``{[$]CC[$],[$]CC(c1ccccc1)[$]}``. The repeat units share the
          same stochastic block (statistical/random sequence).
        * ``"block"`` — several stochastic objects concatenated, e.g.
          ``{[$]CC[$]}{[$]CC(c1ccccc1)[$]}``. Each object is its own block.

        Returns:
            One of ``"homopolymer"``, ``"random"`` or ``"block"``.
        """
        if len(self.repeat_units) <= 1:
            return "homopolymer"
        n_objects = len(
            list(_iter_stochastic_objects(normalize_bigsmiles(self._bigsmiles)))
        )
        return "block" if n_objects > 1 else "random"

    # ------------------------------------------------------------------ #
    # Storage / serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the polymer to a JSON-safe dict.

        The dict is stable and inspectable: it records the source BigSMILES,
        end groups, and per repeat unit the backbone SMILES, sidechain SMILES,
        and descriptor labels. It is sufficient to rebuild the object via
        :meth:`from_dict`.
        """
        return {
            "bigsmiles": self._bigsmiles,
            "num_repeat_units": self.num_repeat_units,
            "is_copolymer": self.is_copolymer,
            "copolymer_type": self.copolymer_type,
            "end_groups": self.end_groups,
            "molecular_weight": self.calculate_molecular_weight(),
            "repeat_units": [unit.to_dict() for unit in self.repeat_units],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BigSmilesPolymer":
        """Rebuild a ``BigSmilesPolymer`` from a serialised dict.

        Reconstruction re-parses the stored ``bigsmiles`` string (the source of
        truth), so classification is regenerated deterministically.

        Args:
            data: A dict produced by :meth:`to_dict`.

        Returns:
            A ``BigSmilesPolymer`` instance.
        """
        bigsmiles_str = data.get("bigsmiles")
        if not bigsmiles_str:
            raise ValueError("Serialised data has no 'bigsmiles' key")
        return cls.from_bigsmiles(bigsmiles_str)

    @classmethod
    def from_json(cls, json_str: str) -> "BigSmilesPolymer":
        """Rebuild a ``BigSmilesPolymer`` from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        return (
            f"BigSmilesPolymer({self._bigsmiles!r}, "
            f"repeat_units={self.num_repeat_units})"
        )
