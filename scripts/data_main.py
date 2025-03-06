import os
import logging
from typing import List, Optional
import fire
import pandas as pd
from pandarallel import pandarallel

from polymetrix.featurizers.polymer import Polymer
from polymetrix.featurizers.chemical_featurizer import (
    NumHBondDonors,
    NumHBondAcceptors,
    NumRotatableBonds,
    NumRings,
    NumNonAromaticRings,
    NumAromaticRings,
    NumAtoms,
    TopologicalSurfaceArea,
    FractionBicyclicRings,
    NumAliphaticHeterocycles,
    SlogPVSA1,
    BalabanJIndex,
    MolecularWeight,
    Sp3CarbonCountFeaturizer,
    Sp2CarbonCountFeaturizer,
    MaxEStateIndex,
    SmrVSA5,
    FpDensityMorgan1,
    HalogenCounts,
    BondCounts,
    BridgingRingsCount,
    MaxRingSize,
    HeteroatomCount,
    HeteroatomDensity,
)
from polymetrix.featurizers.sidechain_backbone_featurizer import (
    SideChainFeaturizer,
    NumSideChainFeaturizer,
    BackBoneFeaturizer,
    NumBackBoneFeaturizer,
    FullPolymerFeaturizer,
    SidechainLengthToStarAttachmentDistanceRatioFeaturizer,
    StarToSidechainMinDistanceFeaturizer,
    SidechainDiversityFeaturizer,
)
from polymetrix.featurizers.multiple_featurizer import MultipleFeaturizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


FULL_POLYMER_FEATURIZERS = [
    (FullPolymerFeaturizer, BalabanJIndex, {}),
    (FullPolymerFeaturizer, NumHBondDonors, {}),
    (FullPolymerFeaturizer, NumHBondAcceptors, {}),
    (FullPolymerFeaturizer, NumRotatableBonds, {}),
    (FullPolymerFeaturizer, NumRings, {}),
    (FullPolymerFeaturizer, NumNonAromaticRings, {}),
    (FullPolymerFeaturizer, NumAromaticRings, {}),
    (FullPolymerFeaturizer, TopologicalSurfaceArea, {}),
    (FullPolymerFeaturizer, FractionBicyclicRings, {}),
    (FullPolymerFeaturizer, NumAliphaticHeterocycles, {}),
    (FullPolymerFeaturizer, SlogPVSA1, {}),
    (FullPolymerFeaturizer, MolecularWeight, {}),
    (FullPolymerFeaturizer, Sp3CarbonCountFeaturizer, {}),
    (FullPolymerFeaturizer, Sp2CarbonCountFeaturizer, {}),
    (FullPolymerFeaturizer, MaxEStateIndex, {}),
    (FullPolymerFeaturizer, SmrVSA5, {}),
    (FullPolymerFeaturizer, FpDensityMorgan1, {}),
    (FullPolymerFeaturizer, HalogenCounts, {}),
    (FullPolymerFeaturizer, BondCounts, {}),
    (FullPolymerFeaturizer, BridgingRingsCount, {}),
    (FullPolymerFeaturizer, MaxRingSize, {}),
    (FullPolymerFeaturizer, HeteroatomDensity, {}),
    (FullPolymerFeaturizer, HeteroatomCount, {}),
]

BACKBONE_LEVEL_FEATURIZERS = [
    (BackBoneFeaturizer, NumAtoms, {}),
    (NumBackBoneFeaturizer, None, {}),
    (BackBoneFeaturizer, BalabanJIndex, {}),
    (BackBoneFeaturizer, NumHBondDonors, {}),
    (BackBoneFeaturizer, NumHBondAcceptors, {}),
    (BackBoneFeaturizer, NumRotatableBonds, {}),
    (BackBoneFeaturizer, NumRings, {}),
    (BackBoneFeaturizer, NumNonAromaticRings, {}),
    (BackBoneFeaturizer, NumAromaticRings, {}),
    (BackBoneFeaturizer, TopologicalSurfaceArea, {}),
    (BackBoneFeaturizer, FractionBicyclicRings, {}),
    (BackBoneFeaturizer, NumAliphaticHeterocycles, {}),
    (BackBoneFeaturizer, SlogPVSA1, {}),
    (BackBoneFeaturizer, MolecularWeight, {}),
    (BackBoneFeaturizer, Sp3CarbonCountFeaturizer, {}),
    (BackBoneFeaturizer, Sp2CarbonCountFeaturizer, {}),
    (BackBoneFeaturizer, MaxEStateIndex, {}),
    (BackBoneFeaturizer, SmrVSA5, {}),
    (BackBoneFeaturizer, FpDensityMorgan1, {}),
    (BackBoneFeaturizer, HalogenCounts, {}),
    (BackBoneFeaturizer, BondCounts, {}),
    (BackBoneFeaturizer, BridgingRingsCount, {}),
    (BackBoneFeaturizer, MaxRingSize, {}),
    (BackBoneFeaturizer, HeteroatomDensity, {}),
    (BackBoneFeaturizer, HeteroatomCount, {}),
]

SIDECHAIN_LEVEL_FEATURIZERS = [
    (SideChainFeaturizer, NumAtoms, {"agg": ["sum", "mean", "max", "min"]}),
    (NumSideChainFeaturizer, None, {}),
    (
        SidechainLengthToStarAttachmentDistanceRatioFeaturizer,
        None,
        {"agg": ["mean", "min", "max", "sum"]},
    ),
    (
        StarToSidechainMinDistanceFeaturizer,
        None,
        {"agg": ["mean", "min", "max", "sum"]},
    ),
    (SidechainDiversityFeaturizer, None, {}),
    (SideChainFeaturizer, BalabanJIndex, {}),
    (SideChainFeaturizer, NumHBondDonors, {}),
    (SideChainFeaturizer, NumHBondAcceptors, {}),
    (SideChainFeaturizer, NumRotatableBonds, {}),
    (SideChainFeaturizer, NumRings, {}),
    (SideChainFeaturizer, NumNonAromaticRings, {}),
    (SideChainFeaturizer, NumAromaticRings, {}),
    (SideChainFeaturizer, TopologicalSurfaceArea, {}),
    (SideChainFeaturizer, FractionBicyclicRings, {}),
    (SideChainFeaturizer, NumAliphaticHeterocycles, {}),
    (SideChainFeaturizer, SlogPVSA1, {}),
    (SideChainFeaturizer, MolecularWeight, {}),
    (SideChainFeaturizer, Sp3CarbonCountFeaturizer, {}),
    (SideChainFeaturizer, Sp2CarbonCountFeaturizer, {}),
    (SideChainFeaturizer, MaxEStateIndex, {}),
    (SideChainFeaturizer, SmrVSA5, {}),
    (SideChainFeaturizer, FpDensityMorgan1, {}),
    (SideChainFeaturizer, HalogenCounts, {}),
    (SideChainFeaturizer, BondCounts, {}),
    (SideChainFeaturizer, BridgingRingsCount, {}),
    (SideChainFeaturizer, MaxRingSize, {}),
    (SideChainFeaturizer, HeteroatomDensity, {}),
    (SideChainFeaturizer, HeteroatomCount, {}),
]


DEFAULT_META_COLUMNS = [
    "polymer",
    "source",
    "tg_range",
    "tg_values",
    "num_of_points",
    "std",
    "reliability",
]

DEFAULT_LABEL_COLUMNS = ["Exp_Tg(K)"]


def create_featurizer_set(featurizer_config: list) -> List[object]:
    """Instantiate featurizers from configuration tuples."""
    featurizers = []
    for container_cls, feature_cls, params in featurizer_config:
        if feature_cls is None:
            featurizers.append(container_cls(**params))
        else:
            inner_feat = feature_cls(**params)
            featurizers.append(container_cls(inner_feat))
    return featurizers


def get_featurizer() -> tuple:
    """Create and configure the MultipleFeaturizer with prefixed feature labels."""
    full_poly = create_featurizer_set(FULL_POLYMER_FEATURIZERS)
    backbone = create_featurizer_set(BACKBONE_LEVEL_FEATURIZERS)
    sidechain = create_featurizer_set(SIDECHAIN_LEVEL_FEATURIZERS)

    def prefix_labels(featurizers: list, prefix: str) -> List[str]:
        return [
            f"{prefix}.{label}" for f in featurizers for label in f.feature_labels()
        ]

    features = (
        sidechain + backbone + full_poly,
        prefix_labels(sidechain, "sidechainlevel.features")
        + prefix_labels(backbone, "backbonelevel.features")
        + prefix_labels(full_poly, "fullpolymerlevel.features"),
    )

    return MultipleFeaturizer(features[0]), features[1]


def compute_features(psmiles: str, featurizer: MultipleFeaturizer) -> pd.Series:
    """Compute features for a single polymer SMILES string."""
    try:
        polymer = Polymer.from_psmiles(psmiles)
        return pd.Series(featurizer.featurize(polymer))
    except Exception as e:
        logger.error(f"Error processing {psmiles}: {str(e)}")
        return pd.Series([None] * len(featurizer.feature_labels()))


def process_dataframe(
    df: pd.DataFrame,
    psmiles_column: str,
    meta_columns: List[str],
    label_columns: List[str],
    output_path: str,
) -> None:
    """Process DataFrame and save results."""
    featurizer, feature_labels = get_featurizer()

    pandarallel.initialize(progress_bar=True)

    features_df = df[psmiles_column].parallel_apply(
        lambda x: compute_features(x, featurizer)
    )
    features_df.columns = feature_labels

    column_renames = {}

    # Handle label columns
    for col in label_columns:
        if col in df.columns:
            column_renames[col] = f"labels.{col}"

    # Handle meta columns
    for col in meta_columns:
        if col in df.columns:
            column_renames[col] = f"meta.{col}"

    # Combine and rename columns
    result_df = pd.concat([df.rename(columns=column_renames), features_df], axis=1)
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")


def main(
    input_path: str,
    output_path: str,
    psmiles_column: str = "PSMILES",
    meta_columns: Optional[List[str]] = None,
    label_columns: Optional[List[str]] = None,
) -> None:
    """Main processing pipeline."""
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    meta_cols = meta_columns or DEFAULT_META_COLUMNS
    label_cols = label_columns or DEFAULT_LABEL_COLUMNS

    df = pd.read_csv(input_path)

    missing_columns = {
        "PSMILES": [psmiles_column],
        "meta": meta_cols,
        "label": label_cols,
    }

    for col_type, columns in missing_columns.items():
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    process_dataframe(
        df=df,
        psmiles_column=psmiles_column,
        meta_columns=meta_cols,
        label_columns=label_cols,
        output_path=output_path,
    )


if __name__ == "__main__":
    fire.Fire(main)