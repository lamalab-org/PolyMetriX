import re
from typing import List, Tuple, Dict, Any

import networkx as nx
from rdkit import Chem
from rdkit.Chem import RWMol, Atom, GetPeriodicTable

try:  # bigsmiles is an optional dependency
    import bigsmiles as _bigsmiles
except ImportError:  # pragma: no cover
    _bigsmiles = None

_PT = GetPeriodicTable()

# BigSMILES bond symbol -> RDKit bond type
_BOND_TYPE_MAP = {
    "": Chem.BondType.SINGLE,
    "-": Chem.BondType.SINGLE,
    "=": Chem.BondType.DOUBLE,
    "#": Chem.BondType.TRIPLE,
    ":": Chem.BondType.AROMATIC,
    "/": Chem.BondType.SINGLE,
    "\\": Chem.BondType.SINGLE,
}


def require_bigsmiles():
    """Return the imported ``bigsmiles`` module, or raise a helpful error."""
    if _bigsmiles is None:  # pragma: no cover
        raise ImportError(
            "The 'bigsmiles' package is required for BigSMILES support. "
            "Install it with `pip install bigsmiles`."
        )
    return _bigsmiles


# A bare bonding descriptor is ``<``, ``>`` or ``$`` optionally followed by an
# index (e.g. ``<1``) and NOT already wrapped in square brackets.
_BARE_DESCRIPTOR = re.compile(r"(?<!\[)([<>$])(\d*)(?!\])")


def _iter_stochastic_objects(text: str):
    """Yield ``(start, end, inner)`` for each top-level ``{...}`` span.

    ``start`` / ``end`` are the indices of the ``{`` and ``}`` characters and
    ``inner`` is the text between them.
    """
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            try:
                j = text.index("}", i)
            except ValueError as exc:
                raise ValueError("Unmatched '{' in BigSMILES string") from exc
            yield i, j, text[i + 1 : j]
            i = j + 1
        else:
            i += 1


def _rebuild(text: str, replacements: Dict[int, Tuple[int, str]]) -> str:
    """Rebuild ``text`` applying ``{start: (end, new_inner)}`` replacements."""
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        if i in replacements:
            j, new_inner = replacements[i]
            out.append("{" + new_inner + "}")
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _expand_simplified(text: str) -> str:
    """Expand the paper's *simplified* form ``{CC}`` -> ``{[$]CC[$]}``.

    In the BigSMILES paper (Lin et al., ACS Cent. Sci. 2019, §2.3) the bonding
    descriptor may be omitted when the repeat unit has exactly two terminal
    connection sites of a single bond type, e.g. polyethylene ``{CC}``, PEG
    ``{CCO}``, polystyrene ``{CC(c1ccccc1)}``. The omitted descriptor is the
    symmetric ``$``, so we restore it at both termini.
    """
    repl: Dict[int, Tuple[int, str]] = {}
    for i, j, inner in _iter_stochastic_objects(text):
        has_desc = any(d in inner for d in ("[<]", "[>]", "[$]", "[]"))
        if inner and not has_desc and "," not in inner:
            repl[i] = (j, "[$]" + inner + "[$]")
    return _rebuild(text, repl)


def _inject_endgroups(text: str) -> str:
    """Add ``[]`` end-group placeholders to *standalone* stochastic objects.

    The parser requires a stochastic object that is not bonded to an external
    atom to carry explicit end-group placeholders, i.e. ``{[$]CC[$]}`` must be
    written ``{[][$]CC[$][]}``. A terminus that is bonded to a real atom (or a
    disconnection ``.``) already has its end group and is left untouched, so
    externally-capped forms such as ``CC{[>][<]CCO[>][<]}O`` pass through
    unchanged.
    """
    repl: Dict[int, Tuple[int, str]] = {}
    n = len(text)
    for i, j, inner in _iter_stochastic_objects(text):
        before = text[i - 1] if i > 0 else ""
        after = text[j + 1] if j + 1 < n else ""
        ext_left = before not in ("", ".")
        ext_right = after not in ("", ".")
        new = inner
        if not ext_left and new[:1] == "[" and not new.startswith("[]"):
            new = "[]" + new
        if not ext_right and new[-1:] == "]" and not new.endswith("[]"):
            new = new + "[]"
        if new != inner:
            repl[i] = (j, new)
    return _rebuild(text, repl)


def normalize_bigsmiles(bigsmiles_str: str) -> str:
    """Normalise any accepted BigSMILES notation into the parser's form.

    The ``bigsmiles`` PyPI parser (v0.0.10) only accepts one very verbose
    spelling: every bonding descriptor bracketed *and* every standalone
    stochastic object carrying explicit empty end-group placeholders, e.g.
    ``{[][<]CCCCO[>][]}``. This helper lets users write the far more readable
    notations from the BigSMILES paper (Lin et al., ACS Cent. Sci. 2019) and
    converts them to that form internally. It accepts:

    * **Paper full form** (bracketed descriptor, no end-group brackets):

      * ``{[$]CC[$]}``            -> ``{[][$]CC[$][]}``
      * ``{[<]CCCCO[>]}``         -> ``{[][<]CCCCO[>][]}``
      * ``{[$]CC(c1ccccc1)[$]}``  -> ``{[][$]CC(c1ccccc1)[$][]}``

    * **Paper simplified form** (descriptor omitted, restored as ``$``):

      * ``{CC}``                  -> ``{[][$]CC[$][]}``
      * ``{CCO}``                 -> ``{[][$]CCO[$][]}``

    * **BigSMILES v1.0 dialect** (bare descriptors, no brackets) — used e.g. by
      the Choi et al. homopolymer dataset:

      * ``{<CCCCO>}``             -> ``{[][<]CCCCO[>][]}``

    * The parser's own verbose form and externally-capped objects such as
      ``CC{[>][<]CCO[>][<]}O`` are returned unchanged, so it is always safe to
      call.

    Args:
        bigsmiles_str: A BigSMILES string in any of the notations above.

    Returns:
        A BigSMILES string in the verbose form the parser accepts.
    """
    text = bigsmiles_str.strip()
    if "{" not in text:
        return text
    # 1) Wrap any bare descriptors (v1.0 dialect): ``<`` -> ``[<]``.
    text = _BARE_DESCRIPTOR.sub(lambda m: f"[{m.group(1)}{m.group(2)}]", text)
    # 2) Restore the omitted descriptor in the simplified form: ``{CC}`` -> ``{[$]CC[$]}``.
    text = _expand_simplified(text)
    # 3) Add empty end-group placeholders to standalone stochastic objects.
    text = _inject_endgroups(text)
    return text


def parse_bigsmiles(bigsmiles_str: str):
    """Parse a BigSMILES string into a ``bigsmiles.BigSMILES`` object.

    The input may be in either the modern bracketed dialect
    (``{[][<]CCO[>][]}``) or the older BigSMILES v1.0 bare-descriptor dialect
    (``{<CCO>}``); the latter is normalised automatically via
    :func:`normalize_bigsmiles`.

    Args:
        bigsmiles_str: The BigSMILES string.

    Returns:
        A parsed ``bigsmiles.BigSMILES`` object.

    Raises:
        ValueError: If the string is empty or cannot be parsed.
    """
    bs_mod = require_bigsmiles()
    if not bigsmiles_str or not isinstance(bigsmiles_str, str):
        raise ValueError("BigSMILES cannot be None or empty")
    normalized = normalize_bigsmiles(bigsmiles_str)
    try:
        return bs_mod.BigSMILES(normalized)
    except Exception as exc:  # bigsmiles raises a variety of error types
        raise ValueError(f"Invalid BigSMILES string: {exc}") from exc


def is_bonding_descriptor(node) -> bool:
    """True if ``node`` is a BigSMILES bonding-descriptor atom ([$]/[<]/[>])."""
    return type(node).__name__ == "BondDescriptorAtom"


def is_atom(node) -> bool:
    """True if ``node`` is a plain BigSMILES atom."""
    return type(node).__name__ == "Atom"


def _walk_atoms_bonds(container, atoms: List, bonds: List) -> Tuple[List, List]:
    """Recursively collect Atom/BondDescriptorAtom and Bond objects.

    Branches (parenthesised groups) nest inside a fragment and are walked
    recursively so ring closures and sidechains are captured.

    Args:
        container: A ``StochasticFragment`` or ``Branch`` exposing ``.nodes``.
        atoms: Accumulator list for atom-like nodes.
        bonds: Accumulator list for bond nodes.

    Returns:
        The ``(atoms, bonds)`` accumulators.
    """
    for node in container.nodes:
        type_name = type(node).__name__
        if type_name in ("Atom", "BondDescriptorAtom"):
            atoms.append(node)
        elif type_name == "Bond":
            bonds.append(node)
        elif type_name == "Branch":
            _walk_atoms_bonds(node, atoms, bonds)
    return atoms, bonds


def _node_key(node) -> Tuple[str, int]:
    """Composite key for a graph node.

    ``Atom.id_`` and ``BondDescriptorAtom.id_`` are numbered independently and
    collide, so we key by ``(type_name, id_)``.
    """
    return (type(node).__name__, node.id_)


def fragment_to_nx(fragment) -> nx.Graph:
    """Convert a BigSMILES ``StochasticFragment`` to a NetworkX graph.

    Bonding-descriptor atoms are emitted with ``element="*"`` so the existing
    ``classify_backbone_and_sidechains`` treats them as connection points. Real
    atoms carry the same node attributes as ``Polymer._mol_to_nx``
    (``atomic_num``, ``element``, ``is_aromatic``, ``formal_charge``) plus the
    bonding-descriptor label where relevant.

    Args:
        fragment: A ``bigsmiles`` ``StochasticFragment`` object.

    Returns:
        A NetworkX graph representing the repeat unit.
    """
    graph = nx.Graph()
    atoms, bonds = _walk_atoms_bonds(fragment, [], [])
    # Ring-closure bonds are stored on ``fragment.rings`` rather than in the
    # ``.nodes`` walk, so add them explicitly (otherwise aromatic/aliphatic
    # rings come out as open chains and lose a bond).
    for ring_bond in getattr(fragment, "rings", []) or []:
        if ring_bond not in bonds:
            bonds.append(ring_bond)

    for node in atoms:
        key = _node_key(node)
        if is_bonding_descriptor(node):
            graph.add_node(
                key,
                element="*",
                atomic_num=0,
                is_aromatic=False,
                formal_charge=0,
                descriptor=str(node.descriptor),
            )
        else:
            graph.add_node(
                key,
                element=node.symbol,
                atomic_num=_PT.GetAtomicNumber(node.symbol),
                is_aromatic=bool(node.aromatic),
                formal_charge=int(node.charge),
            )

    for bond in bonds:
        graph.add_edge(
            _node_key(bond.atom1),
            _node_key(bond.atom2),
            bond_type=_BOND_TYPE_MAP.get(bond.symbol, Chem.BondType.SINGLE),
            bigsmiles_symbol=bond.symbol,
        )
    return graph


def stochastic_fragment_graphs(bigsmiles_obj) -> List[Tuple[Any, nx.Graph]]:
    """Enumerate every repeat unit (stochastic fragment) as a graph.

    A BigSMILES string may contain several stochastic objects, and each object
    may hold several fragments (a copolymer). Each fragment is one repeat unit.

    Args:
        bigsmiles_obj: A parsed ``bigsmiles.BigSMILES`` object.

    Returns:
        A list of ``(fragment, graph)`` tuples, in document order.
    """
    results = []
    for node in bigsmiles_obj.nodes:
        if type(node).__name__ == "StochasticObject":
            for fragment in node.nodes:
                if type(fragment).__name__ == "StochasticFragment":
                    results.append((fragment, fragment_to_nx(fragment)))
    return results


def repeat_unit_graphs(bigsmiles_str: str) -> List[Tuple[Any, nx.Graph]]:
    """Enumerate every repeat unit of a BigSMILES string as a graph.

    This is the string-level entry point used by :class:`BigSmilesPolymer`. It
    distinguishes the two ways copolymers are written in the BigSMILES paper
    (Lin et al., *ACS Cent. Sci.* 2019):

    * **Random copolymer** — one stochastic object with comma-separated
      fragments, e.g. ``{[$]CC[$],[$]CC(c1ccccc1)[$]}``.
    * **Block copolymer** — direct concatenation of stochastic objects, e.g.
      ``{[$]CC[$]}{[$]CC(c1ccccc1)[$]}`` (Figure 4c).

    When several objects are concatenated, the parser fuses the two facing
    bonding descriptors at each block junction into a single inter-block bond,
    which would leave the flanking repeat units with only one ``*`` connection
    point and break backbone tracing. To classify each block on its own terms,
    every top-level ``{...}`` object is parsed **standalone** (which re-caps it
    with ``[]`` end groups and restores both connection points). A single
    object — the homopolymer and random-copolymer case — is parsed once,
    unchanged.

    Args:
        bigsmiles_str: A raw BigSMILES string (any accepted notation).

    Returns:
        A list of ``(fragment, graph)`` tuples, one per repeat unit, in
        document order.
    """
    normalized = normalize_bigsmiles(bigsmiles_str)
    objects = list(_iter_stochastic_objects(normalized))
    if not objects:
        return []
    # Parse every stochastic object standalone. This is uniformly robust:
    #   * homopolymers and random copolymers (comma-separated fragments inside
    #     one object) parse as-is;
    #   * externally end-capped objects (e.g. ``CC{[$]...[$]}Br`` from an ATRP
    #     or RAFT polymer) drop the caps, which the parser needs to treat the
    #     object on its own;
    #   * block copolymers (concatenated objects) avoid the junction
    #     descriptor fusion that would otherwise strip a connection point from
    #     each interior block.
    # ``inner`` already carries the object's own end-group placeholders after
    # normalisation, so ``{ + inner + }`` is a valid standalone object.
    results = []
    for _, _, inner in objects:
        standalone = "{" + inner + "}"
        results.extend(stochastic_fragment_graphs(parse_bigsmiles(standalone)))
    return results


def end_group_atoms(bigsmiles_obj) -> List[str]:
    """Return element symbols of explicit end-group atoms (top-level atoms).

    In ``CC{[>][<]CCO[>][<]}O`` the leading ``CC`` and trailing ``O`` are
    explicit end groups living outside the stochastic object.

    Args:
        bigsmiles_obj: A parsed ``bigsmiles.BigSMILES`` object.

    Returns:
        A list of element symbols for the top-level (end-group) atoms.
    """
    return [
        node.symbol for node in bigsmiles_obj.nodes if type(node).__name__ == "Atom"
    ]


def graph_to_mol(graph: nx.Graph, node_keys: List[Tuple[str, int]]) -> Chem.Mol:
    """Build an RDKit molecule from a subset of graph nodes.

    Bonding-descriptor nodes (``element="*"``) become dummy atoms (atomic
    number 0, RDKit ``*``), exactly mirroring the ``*``-terminated fragments
    that the pSMILES ``Polymer._extract_substructure_mol`` produces, so the
    descriptor-based featurizers behave identically.

    Args:
        graph: The full repeat-unit graph.
        node_keys: The node keys to include in the molecule.

    Returns:
        An RDKit ``Mol`` (possibly empty).
    """
    if not node_keys:
        return Chem.MolFromSmiles("")

    mol = RWMol()
    key_to_idx: Dict[Tuple[str, int], int] = {}
    node_set = set(node_keys)

    for key in node_keys:
        data = graph.nodes[key]
        atom = Atom(int(data.get("atomic_num", 0)))
        atom.SetFormalCharge(int(data.get("formal_charge", 0)))
        if data.get("is_aromatic"):
            atom.SetIsAromatic(True)
        key_to_idx[key] = mol.AddAtom(atom)

    for u, v, edata in graph.edges(data=True):
        if u in node_set and v in node_set:
            mol.AddBond(
                key_to_idx[u],
                key_to_idx[v],
                edata.get("bond_type", Chem.BondType.SINGLE),
            )
    return mol.GetMol()


def mol_to_smiles(mol: Chem.Mol) -> str:
    """Be canonical SMILES for a reconstructed fragment mol.

    Sanitisation is attempted but not required; on failure the unsanitised
    SMILES is returned so serialisation never crashes on an odd fragment.

    Args:
        mol: An RDKit molecule.

    Returns:
        A SMILES string (empty string for an empty mol).
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return ""
    try:
        work = Chem.Mol(mol)
        Chem.SanitizeMol(work)
        return Chem.MolToSmiles(work)
    except Exception:
        try:
            return Chem.MolToSmiles(mol, canonical=False)
        except Exception:
            return ""
