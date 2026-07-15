import networkx as nx
import pytest
from rdkit import Chem

pytest.importorskip("bigsmiles")

from polymetrix.bigsmiles import BigSmilesPolymer, RepeatUnit
from polymetrix.featurizers.polymer import Polymer
from polymetrix.featurizers.sidechain_backbone_featurizer import (
    SideChainFeaturizer,
    NumSideChainFeaturizer,
    BackBoneFeaturizer,
    FullPolymerFeaturizer,
)
from polymetrix.featurizers.chemical_featurizer import NumAtoms, MolecularWeight


# --- parsing -------------------------------------------------------------- #
def test_bigsmiles_creation():
    polymer = BigSmilesPolymer.from_bigsmiles("{[][<]CC(c1ccccc1)[>][]}")
    assert isinstance(polymer, BigSmilesPolymer)
    assert polymer.bigsmiles == "{[][<]CC(c1ccccc1)[>][]}"
    assert isinstance(polymer.graph, nx.Graph)
    assert polymer.num_repeat_units == 1
    assert not polymer.is_copolymer


def test_invalid_bigsmiles():
    with pytest.raises(ValueError):
        BigSmilesPolymer.from_bigsmiles("not a bigsmiles")


def test_empty_bigsmiles():
    with pytest.raises(ValueError):
        BigSmilesPolymer.from_bigsmiles("")


def test_no_repeat_unit_raises():
    # A plain SMILES with no stochastic object has no repeat unit.
    with pytest.raises(ValueError):
        BigSmilesPolymer.from_bigsmiles("CCO")


# --- classification ------------------------------------------------------- #
def test_polystyrene_classification():
    polymer = BigSmilesPolymer.from_bigsmiles("{[][<]CC(c1ccccc1)[>][]}")
    unit = polymer.repeat_units[0]
    assert isinstance(unit, RepeatUnit)
    # one aromatic-ring sidechain
    assert len(unit.sidechain_molecules) == 1
    assert Chem.MolToSmiles(_sanitize(unit.sidechain_molecules[0])) == "c1ccccc1"


def test_backbone_only_polymer_has_no_sidechain():
    polymer = BigSmilesPolymer.from_bigsmiles("{[][<]CCO[>][]}")  # PEG
    _, sidechains = polymer.get_backbone_and_sidechain_molecules()
    assert all(m.GetNumAtoms() == 0 for m in sidechains) or sidechains == []


def test_dollar_descriptor_pdms():
    polymer = BigSmilesPolymer.from_bigsmiles("{[][$]O[Si](C)(C)[$][]}")
    assert polymer.num_repeat_units == 1
    assert polymer.repeat_units[0].descriptors == ["[$]", "[$]"]


def test_copolymer_has_two_repeat_units():
    polymer = BigSmilesPolymer.from_bigsmiles(
        "{[][<]CC(c1ccccc1)[>],[<]CC(C(=O)OC)(C)[>][]}"
    )
    assert polymer.num_repeat_units == 2
    assert polymer.is_copolymer
    backbones, sidechains = polymer.get_backbone_and_sidechain_molecules()
    assert len(backbones) == 2
    assert len(sidechains) == 2


# --- interface parity with Polymer --------------------------------------- #
def test_get_backbone_and_sidechain_molecules_shape():
    polymer = BigSmilesPolymer.from_bigsmiles("{[][<]CC(c1ccccc1)[>][]}")
    backbone, sidechains = polymer.get_backbone_and_sidechain_molecules()
    assert isinstance(backbone[0], Chem.Mol)
    assert isinstance(sidechains[0], Chem.Mol)


def test_get_backbone_and_sidechain_graphs():
    polymer = BigSmilesPolymer.from_bigsmiles("{[][<]CC(c1ccccc1)[>][]}")
    backbones, sidechains = polymer.get_backbone_and_sidechain_graphs()
    assert isinstance(backbones[0], nx.Graph)
    assert isinstance(sidechains[0], nx.Graph)


def test_psmiles_bridge():
    polymer = BigSmilesPolymer.from_bigsmiles("{[][<]CC(c1ccccc1)[>][]}")
    # canonical pSMILES of polystyrene repeat unit
    assert Chem.CanonSmiles(polymer.psmiles) == Chem.CanonSmiles("*CC(*)c1ccccc1")


def test_molecular_weight_positive():
    polymer = BigSmilesPolymer.from_bigsmiles("{[][<]CC(c1ccccc1)[>][]}")
    assert polymer.calculate_molecular_weight() > 0


# --- featurizer integration (unmodified featurizers) --------------------- #
@pytest.mark.parametrize(
    "featurizer",
    [
        NumSideChainFeaturizer(),
        BackBoneFeaturizer(NumAtoms()),
        SideChainFeaturizer(NumAtoms()),
        FullPolymerFeaturizer(NumAtoms()),
        FullPolymerFeaturizer(MolecularWeight()),
    ],
)
def test_featurizers_run_on_bigsmiles(featurizer):
    polymer = BigSmilesPolymer.from_bigsmiles("{[][<]CC(c1ccccc1)[>][]}")
    result = featurizer.featurize(polymer)
    assert result is not None
    assert len(result) >= 1


# --- parity with the pSMILES Polymer ------------------------------------- #
@pytest.mark.parametrize(
    "psmiles,bigsmiles_str",
    [
        ("[*]CC[*]", "{[][<]CC[>][]}"),
        ("[*]CC([*])c1ccccc1", "{[][<]CC(c1ccccc1)[>][]}"),
        ("[*]CC([*])(C)C(=O)OC", "{[][<]CC(C(=O)OC)(C)[>][]}"),
        ("[*]CCO[*]", "{[][<]CCO[>][]}"),
        ("[*]CC([*])Cl", "{[][<]CC(Cl)[>][]}"),
    ],
)
def test_parity_with_psmiles(psmiles, bigsmiles_str):
    p = Polymer.from_psmiles(psmiles)
    b = BigSmilesPolymer.from_bigsmiles(bigsmiles_str)
    _, p_sc = p.get_backbone_and_sidechain_molecules()
    _, b_sc = b.get_backbone_and_sidechain_molecules()
    p_smis = sorted(_smiles(m) for m in p_sc if m.GetNumAtoms())
    b_smis = sorted(_smiles(m) for m in b_sc if m.GetNumAtoms())
    assert p_smis == b_smis


# --- older-dialect (BigSMILES v1.0) normalisation ------------------------ #
from polymetrix.bigsmiles.bigsmiles_utils import normalize_bigsmiles


@pytest.mark.parametrize(
    "old_dialect,expected",
    [
        ("{<CCCCO>}", "{[][<]CCCCO[>][]}"),
        ("{<N[Si](C)(C)>}", "{[][<]N[Si](C)(C)[>][]}"),
        ("{$CC=C(CCCC)C$}", "{[][$]CC=C(CCCC)C[$][]}"),
        ("{$CC(OCCCCCCCC)$}", "{[][$]CC(OCCCCCCCC)[$][]}"),
    ],
)
def test_normalize_old_dialect(old_dialect, expected):
    assert normalize_bigsmiles(old_dialect) == expected


@pytest.mark.parametrize(
    "modern",
    [
        "{[][<]CCO[>][]}",
        "{[][<]CC(c1ccccc1)[>][]}",
        "{[][$]O[Si](C)(C)[$][]}",
    ],
)
def test_normalize_modern_dialect_unchanged(modern):
    assert normalize_bigsmiles(modern) == modern


@pytest.mark.parametrize(
    "old_dialect",
    ["{<CCCCO>}", "{<N[Si](C)(C)>}", "{$CC=C(CCCC)C$}", "{$CC(OCCCCCCCC)$}"],
)
def test_old_dialect_parses_and_classifies(old_dialect):
    polymer = BigSmilesPolymer.from_bigsmiles(old_dialect)
    assert polymer.num_repeat_units == 1
    # backbone must contain the two connection points
    assert polymer.repeat_units[0].backbone_smiles.count("*") == 2


# --- serialisation -------------------------------------------------------- #
def test_to_dict_structure():
    polymer = BigSmilesPolymer.from_bigsmiles("{[][<]CC(c1ccccc1)[>][]}")
    d = polymer.to_dict()
    assert d["bigsmiles"] == "{[][<]CC(c1ccccc1)[>][]}"
    assert d["num_repeat_units"] == 1
    assert d["repeat_units"][0]["sidechain_smiles"] == ["c1ccccc1"]


def test_json_round_trip():
    polymer = BigSmilesPolymer.from_bigsmiles(
        "{[][<]CC(c1ccccc1)[>],[<]CC(C(=O)OC)(C)[>][]}"
    )
    restored = BigSmilesPolymer.from_json(polymer.to_json())
    assert restored.to_dict() == polymer.to_dict()


def test_from_dict_round_trip():
    polymer = BigSmilesPolymer.from_bigsmiles("{[][<]CCO[>][]}")
    restored = BigSmilesPolymer.from_dict(polymer.to_dict())
    assert restored.bigsmiles == polymer.bigsmiles
    assert restored.to_dict() == polymer.to_dict()


# --- paper notation (ACS Cent. Sci. 2019) --------------------------------- #
@pytest.mark.parametrize(
    "paper,expected",
    [
        # full form: bracketed descriptor, no end-group brackets
        ("{[$]CC[$]}", "{[][$]CC[$][]}"),
        ("{[<]CCCCO[>]}", "{[][<]CCCCO[>][]}"),
        ("{[$]CC(c1ccccc1)[$]}", "{[][$]CC(c1ccccc1)[$][]}"),
        # simplified form: descriptor omitted, restored as $
        ("{CC}", "{[][$]CC[$][]}"),
        ("{CCO}", "{[][$]CCO[$][]}"),
        ("{CC(c1ccccc1)}", "{[][$]CC(c1ccccc1)[$][]}"),
        # copolymer, full form
        (
            "{[$]CC(c1ccccc1)[$],[$]CC(C(=O)OC)(C)[$]}",
            "{[][$]CC(c1ccccc1)[$],[$]CC(C(=O)OC)(C)[$][]}",
        ),
    ],
)
def test_normalize_paper_notation(paper, expected):
    assert normalize_bigsmiles(paper) == expected


def test_externally_capped_object_unchanged():
    # a stochastic object bonded to real end atoms already has its end groups
    capped = "CC{[>][<]CCO[>][<]}O"
    assert normalize_bigsmiles(capped) == capped


@pytest.mark.parametrize(
    "paper,backbone,sidechains",
    [
        ("{CC}", "*CC*", []),
        ("{CCO}", "*CCO*", []),
        ("{[$]CC(c1ccccc1)[$]}", "*CC*", ["c1ccccc1"]),
        ("{[$]CC(C(=O)OC)(C)[$]}", "*CC(*)C", ["COC=O"]),
        ("{[$]O[Si](C)(C)[$]}", "*O[Si](*)(C)C", []),
    ],
)
def test_paper_notation_classifies(paper, backbone, sidechains):
    polymer = BigSmilesPolymer.from_bigsmiles(paper)
    unit = polymer.repeat_units[0]
    assert unit.backbone_smiles == backbone
    assert unit.sidechain_smiles == sidechains


@pytest.mark.parametrize(
    "paper_form,verbose_form",
    [
        ("{CC}", "{[][<]CC[>][]}"),
        ("{[$]CC(c1ccccc1)[$]}", "{[][<]CC(c1ccccc1)[>][]}"),
        ("{[$]CC(C(=O)OC)(C)[$]}", "{[][<]CC(C(=O)OC)(C)[>][]}"),
    ],
)
def test_paper_and_verbose_forms_agree(paper_form, verbose_form):
    """The clean paper notation classifies identically to the verbose form."""
    a = BigSmilesPolymer.from_bigsmiles(paper_form)
    b = BigSmilesPolymer.from_bigsmiles(verbose_form)
    assert a.repeat_units[0].backbone_smiles == b.repeat_units[0].backbone_smiles
    assert a.repeat_units[0].sidechain_smiles == b.repeat_units[0].sidechain_smiles
    assert abs(a.calculate_molecular_weight() - b.calculate_molecular_weight()) < 1e-6


# --- copolymer architecture (random vs block) ---------------------------- #
@pytest.mark.parametrize(
    "bigsmiles_str, expected_type, n_units",
    [
        ("{[$]CC(c1ccccc1)[$]}", "homopolymer", 1),
        ("{[$]CC(c1ccccc1)[$],[$]CC(C(=O)OC)(C)[$]}", "random", 2),
        ("{[$]CC[$]}{[$]CC(c1ccccc1)[$]}", "block", 2),
        ("{[$]CC[$]}{[$]CCO[$]}{[$]CC(c1ccccc1)[$]}", "block", 3),
    ],
)
def test_copolymer_type(bigsmiles_str, expected_type, n_units):
    """Random (comma) and block (concatenated) copolymers are distinguished."""
    p = BigSmilesPolymer.from_bigsmiles(bigsmiles_str)
    assert p.copolymer_type == expected_type
    assert p.num_repeat_units == n_units
    assert p.is_copolymer == (n_units > 1)


@pytest.mark.parametrize(
    "bigsmiles_str, backbones",
    [
        # Block copolymer: each block classified independently, both stars kept.
        ("{[$]CC[$]}{[$]CC(c1ccccc1)[$]}", ["*CC*", "*CC*"]),
        # Triblock PE-b-PEG-b-PS.
        (
            "{[$]CC[$]}{[$]CCO[$]}{[$]CC(c1ccccc1)[$]}",
            ["*CC*", "*CCO*", "*CC*"],
        ),
    ],
)
def test_block_copolymer_backbones(bigsmiles_str, backbones):
    """Block copolymers yield non-empty backbones for every block.

    Regression test: concatenated stochastic objects fuse their facing bonding
    descriptors at each junction, which previously left interior blocks with a
    single connection point and produced empty backbones.
    """
    p = BigSmilesPolymer.from_bigsmiles(bigsmiles_str)
    assert [u.backbone_smiles for u in p.repeat_units] == backbones
    for u in p.repeat_units:
        assert u.backbone_smiles, "backbone must not be empty"


def test_block_copolymer_serialization_round_trip():
    """Block copolymers serialise and reconstruct, preserving copolymer_type."""
    p = BigSmilesPolymer.from_bigsmiles("{[$]CC[$]}{[$]CC(c1ccccc1)[$]}")
    restored = BigSmilesPolymer.from_json(p.to_json())
    assert restored.to_dict() == p.to_dict()
    assert restored.copolymer_type == "block"


@pytest.mark.parametrize(
    "bigsmiles_str, copoly_type, backbones",
    [
        # SI Example 9: polystyrene from ATRP, 1-phenylethyl-bromide initiator
        # and Br end group flanking the stochastic object.
        ("CC(c1ccccc1){[$]CC(c1ccccc1)[$]}Br", "homopolymer", ["*CC*"]),
        # Conjugate-descriptor object with external end groups.
        ("CC{[<]CCO[>]}O", "homopolymer", ["*CCO*"]),
        # External end groups flanking two concatenated blocks.
        (
            "CCC(C){[$]CC(c1ccccc1)[$]}{[$]CCO[$]}[H]",
            "block",
            ["*CC*", "*CCO*"],
        ),
    ],
)
def test_externally_capped_polymers(bigsmiles_str, copoly_type, backbones):
    """Polymers with explicit external end groups classify correctly.

    Each stochastic object is parsed standalone, so the flanking end groups
    (initiator / terminator, as written for real ATRP and RAFT polymers in the
    BigSMILES SI) do not prevent the object from being recognised.
    """
    p = BigSmilesPolymer.from_bigsmiles(bigsmiles_str)
    assert p.copolymer_type == copoly_type
    assert [u.backbone_smiles for u in p.repeat_units] == backbones


# --- helpers -------------------------------------------------------------- #
def _sanitize(mol):
    m = Chem.Mol(mol)
    Chem.SanitizeMol(m)
    return m


def _smiles(mol):
    try:
        return Chem.MolToSmiles(_sanitize(mol))
    except Exception:
        return Chem.MolToSmiles(mol, canonical=False)
