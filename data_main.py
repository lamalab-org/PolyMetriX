import os
import logging
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
    MolecularWeightFeaturizer,
    Sp3CarbonCountFeaturizer,
    Sp2CarbonCountFeaturizer,
    MaxEStateIndex,
    SMR_VSA5,
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
    (FullPolymerFeaturizer, MolecularWeightFeaturizer, {}),
    (FullPolymerFeaturizer, Sp3CarbonCountFeaturizer, {}),
    (FullPolymerFeaturizer, Sp2CarbonCountFeaturizer, {}),
    (FullPolymerFeaturizer, MaxEStateIndex, {}),
    (FullPolymerFeaturizer, SMR_VSA5, {}),
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
    (BackBoneFeaturizer, MolecularWeightFeaturizer, {}),
    (BackBoneFeaturizer, Sp3CarbonCountFeaturizer, {}),
    (BackBoneFeaturizer, Sp2CarbonCountFeaturizer, {}),
    (BackBoneFeaturizer, MaxEStateIndex, {}),
    (BackBoneFeaturizer, SMR_VSA5, {}),
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
    (SideChainFeaturizer, MolecularWeightFeaturizer, {}),
    (SideChainFeaturizer, Sp3CarbonCountFeaturizer, {}),
    (SideChainFeaturizer, Sp2CarbonCountFeaturizer, {}),
    (SideChainFeaturizer, MaxEStateIndex, {}),
    (SideChainFeaturizer, SMR_VSA5, {}),
    (SideChainFeaturizer, FpDensityMorgan1, {}),
    (SideChainFeaturizer, HalogenCounts, {}),
    (SideChainFeaturizer, BondCounts, {}),
    (SideChainFeaturizer, BridgingRingsCount, {}),
    (SideChainFeaturizer, MaxRingSize, {}),
    (SideChainFeaturizer, HeteroatomDensity, {}),
    (SideChainFeaturizer, HeteroatomCount, {}),
]


def instantiate_featurizers(features):
    """Instantiates a list of featurizers from full, backbone, or sidechain level configurations."""
    featurizers = []
    for featurizer_class, feature_class, args in features:
        if feature_class is None:
            # Standalone featurizer (e.g., NumSideChainFeaturizer)
            featurizers.append(featurizer_class(**args))
        else:
            # Wrapped featurizer (e.g., SideChainFeaturizer(NumAtoms(...)))
            featurizers.append(featurizer_class(feature_class(**args)))
    return featurizers


def create_featurizer():
    """Creates and configures a MultipleFeaturizer with various polymer feature extractors."""
    full_poly_extractors = instantiate_featurizers(FULL_POLYMER_FEATURIZERS)
    backbone_extractors = instantiate_featurizers(BACKBONE_LEVEL_FEATURIZERS)
    sidechain_extractors = instantiate_featurizers(SIDECHAIN_LEVEL_FEATURIZERS)

    # Combine all featurizers into a single MultipleFeaturizer
    all_extractors = sidechain_extractors + backbone_extractors + full_poly_extractors
    return MultipleFeaturizer(all_extractors)


def calculate_features(psmiles, featurizer):
    """Calculates features for a given polymer SMILES string using the specified featurizer."""
    try:
        polymer_instance = Polymer.from_psmiles(psmiles)
        features = featurizer.featurize(polymer_instance)
        return pd.Series(features)
    except Exception as e:
        logger.error(f"Error processing PSMILES {psmiles}: {str(e)}")
        return pd.Series([None] * len(featurizer.feature_labels()))


def process_csv(input_file, output_file, psmiles_column):
    """Processes a CSV file containing polymer SMILES and calculates their features."""
    pandarallel.initialize(progress_bar=True)
    df = pd.read_csv(input_file)
    featurizer = create_featurizer()

    feature_df = df[psmiles_column].parallel_apply(
        lambda x: calculate_features(x, featurizer)
    )
    feature_df.columns = featurizer.feature_labels()
    result_df = pd.concat([df, feature_df], axis=1)

    result_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")


def main(input_path, output_path, PSMILES_COLUMN="PSMILES"):
    """Main function to process polymer features from an input CSV file."""
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info("Please ensure the input file exists and the path is correct.")
        return

    process_csv(input_path, output_path, PSMILES_COLUMN)


if __name__ == "__main__":
    fire.Fire(main)
