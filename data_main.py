import os

import fire
import pandas as pd
from pandarallel import pandarallel

from polymetrix.featurizer import (
    BalabanJIndex,
    BackBoneFeaturizer,
    BondCounts,
    BridgingRingsCount,
    FpDensityMorgan1,
    FractionBicyclicRings,
    FullPolymerFeaturizer,
    HalogenCounts,
    HeteroatomCount,
    HeteroatomDensity,
    MaxEStateIndex,
    MaxRingSize,
    MolecularWeightFeaturizer,
    MultipleFeaturizer,
    NumAliphaticHeterocycles,
    NumAromaticRings,
    NumAtoms,
    NumBackBoneFeaturizer,
    NumHBondAcceptors,
    NumHBondDonors,
    NumNonAromaticRings,
    NumRings,
    NumRotatableBonds,
    NumSideChainFeaturizer,
    SMR_VSA5,
    SideChainFeaturizer,
    SidechainDiversityFeaturizer,
    SidechainLengthToStarAttachmentDistanceRatioFeaturizer,
    SlogPVSA1,
    Sp2CarbonCountFeaturizer,
    Sp3CarbonCountFeaturizer,
    StarToSidechainMinDistanceFeaturizer,
    TopologicalSurfaceArea,
)
from polymetrix.polymer import Polymer


def create_featurizer():
    """Creates and configures a MultipleFeaturizer with various polymer feature extractors.

    Returns:
        MultipleFeaturizer: A configured featurizer combining multiple polymer feature extractors.
    """
    sidechain_length = SideChainFeaturizer(NumAtoms(agg=["sum", "mean", "max", "min"]))
    num_sidechains = NumSideChainFeaturizer()
    backbone_length = BackBoneFeaturizer(NumAtoms())
    num_backbones = NumBackBoneFeaturizer()
    hbond_donors = FullPolymerFeaturizer(NumHBondDonors())
    hbond_acceptors = FullPolymerFeaturizer(NumHBondAcceptors())
    rotatable_bonds = FullPolymerFeaturizer(NumRotatableBonds())
    rings = FullPolymerFeaturizer(NumRings())
    non_aromatic_rings = FullPolymerFeaturizer(NumNonAromaticRings())
    aromatic_rings = FullPolymerFeaturizer(NumAromaticRings())
    topological_surface_area = FullPolymerFeaturizer(TopologicalSurfaceArea())
    fraction_bicyclic_rings = FullPolymerFeaturizer(FractionBicyclicRings())
    num_aliphatic_heterocycles = FullPolymerFeaturizer(NumAliphaticHeterocycles())
    slogpvsa1 = FullPolymerFeaturizer(SlogPVSA1())
    balabanjindex = FullPolymerFeaturizer(BalabanJIndex())
    molecular_weight = FullPolymerFeaturizer(MolecularWeightFeaturizer())
    sp3_carbon_count = FullPolymerFeaturizer(Sp3CarbonCountFeaturizer())
    sp2_carbon_count = FullPolymerFeaturizer(Sp2CarbonCountFeaturizer())
    max_estate_index = FullPolymerFeaturizer(MaxEStateIndex())
    smr_vsa5 = FullPolymerFeaturizer(SMR_VSA5())
    fp_density_morgan1 = FullPolymerFeaturizer(FpDensityMorgan1())
    halogen_counts = FullPolymerFeaturizer(HalogenCounts())
    bond_counts = FullPolymerFeaturizer(BondCounts())
    bridging_rings_count = FullPolymerFeaturizer(BridgingRingsCount())
    max_ring_size = FullPolymerFeaturizer(MaxRingSize())
    heteroatom_density = FullPolymerFeaturizer(HeteroatomDensity())
    heteroatom_count = FullPolymerFeaturizer(HeteroatomCount())

    sidechain_length_to_star_attachment_distance_ratio = (
        SidechainLengthToStarAttachmentDistanceRatioFeaturizer(
            agg=["mean", "min", "max", "sum"]
        )
    )
    star_to_sidechain_min_distance = StarToSidechainMinDistanceFeaturizer(
        agg=["mean", "min", "max", "sum"]
    )
    sidechain_diversity = SidechainDiversityFeaturizer()
    backbone_balabanjindex = BackBoneFeaturizer(BalabanJIndex())
    backbone_hbond_donors = BackBoneFeaturizer(NumHBondDonors())
    backbone_hbond_acceptors = BackBoneFeaturizer(NumHBondAcceptors())
    backbone_rotatable_bonds = BackBoneFeaturizer(NumRotatableBonds())
    backbone_rings = BackBoneFeaturizer(NumRings())
    backbone_non_aromatic_rings = BackBoneFeaturizer(NumNonAromaticRings())
    backbone_aromatic_rings = BackBoneFeaturizer(NumAromaticRings())
    backbone_topological_surface_area = BackBoneFeaturizer(TopologicalSurfaceArea())
    backbone_fraction_bicyclic_rings = BackBoneFeaturizer(FractionBicyclicRings())
    backbone_num_aliphatic_heterocycles = BackBoneFeaturizer(NumAliphaticHeterocycles())
    backbone_slogpvsa1 = BackBoneFeaturizer(SlogPVSA1())
    backbone_molecular_weight = BackBoneFeaturizer(MolecularWeightFeaturizer())
    backbone_sp3_carbon_count = BackBoneFeaturizer(Sp3CarbonCountFeaturizer())
    backbone_sp2_carbon_count = BackBoneFeaturizer(Sp2CarbonCountFeaturizer())
    backbone_max_estate_index = BackBoneFeaturizer(MaxEStateIndex())
    backbone_smr_vsa5 = BackBoneFeaturizer(SMR_VSA5())
    backbone_fp_density_morgan1 = BackBoneFeaturizer(FpDensityMorgan1())
    backbone_halogen_counts = BackBoneFeaturizer(HalogenCounts())
    backbone_bond_counts = BackBoneFeaturizer(BondCounts())
    backbone_bridging_rings_count = BackBoneFeaturizer(BridgingRingsCount())
    backbone_max_ring_size = BackBoneFeaturizer(MaxRingSize())
    backbone_heteroatom_density = BackBoneFeaturizer(HeteroatomDensity())
    backbone_heteroatom_count = BackBoneFeaturizer(HeteroatomCount())
    sidechain_balabanjindex = SideChainFeaturizer(BalabanJIndex())
    sidechain_hbond_donors = SideChainFeaturizer(NumHBondDonors())
    sidechain_hbond_acceptors = SideChainFeaturizer(NumHBondAcceptors())
    sidechain_rotatable_bonds = SideChainFeaturizer(NumRotatableBonds())
    sidechain_rings = SideChainFeaturizer(NumRings())
    sidechain_non_aromatic_rings = SideChainFeaturizer(NumNonAromaticRings())
    sidechain_aromatic_rings = SideChainFeaturizer(NumAromaticRings())
    sidechain_topological_surface_area = SideChainFeaturizer(TopologicalSurfaceArea())
    sidechain_fraction_bicyclic_rings = SideChainFeaturizer(FractionBicyclicRings())
    sidechain_num_aliphatic_heterocycles = SideChainFeaturizer(
        NumAliphaticHeterocycles()
    )
    sidechain_slogpvsa1 = SideChainFeaturizer(SlogPVSA1())
    sidechain_molecular_weight = SideChainFeaturizer(MolecularWeightFeaturizer())
    sidechain_sp3_carbon_count = SideChainFeaturizer(Sp3CarbonCountFeaturizer())
    sidechain_sp2_carbon_count = SideChainFeaturizer(Sp2CarbonCountFeaturizer())
    sidechain_max_estate_index = SideChainFeaturizer(MaxEStateIndex())
    sidechain_smr_vsa5 = SideChainFeaturizer(SMR_VSA5())
    sidechain_fp_density_morgan1 = SideChainFeaturizer(FpDensityMorgan1())
    sidechain_halogen_counts = SideChainFeaturizer(HalogenCounts())
    sidechain_bond_counts = SideChainFeaturizer(BondCounts())
    sidechain_bridging_rings_count = SideChainFeaturizer(BridgingRingsCount())
    sidechain_max_ring_size = SideChainFeaturizer(MaxRingSize())
    sidechain_heteroatom_density = SideChainFeaturizer(HeteroatomDensity())
    sidechain_heteroatom_count = SideChainFeaturizer(HeteroatomCount())

    return MultipleFeaturizer(
        [
            sidechain_length,
            num_sidechains,
            backbone_length,
            num_backbones,
            hbond_donors,
            hbond_acceptors,
            rotatable_bonds,
            rings,
            non_aromatic_rings,
            aromatic_rings,
            topological_surface_area,
            fraction_bicyclic_rings,
            num_aliphatic_heterocycles,
            slogpvsa1,
            balabanjindex,
            molecular_weight,
            sp3_carbon_count,
            sp2_carbon_count,
            max_estate_index,
            smr_vsa5,
            fp_density_morgan1,
            halogen_counts,
            bond_counts,
            bridging_rings_count,
            max_ring_size,
            heteroatom_density,
            heteroatom_count,
            sidechain_length_to_star_attachment_distance_ratio,
            star_to_sidechain_min_distance,
            sidechain_diversity,
            backbone_balabanjindex,
            backbone_hbond_donors,
            backbone_hbond_acceptors,
            backbone_rotatable_bonds,
            backbone_rings,
            backbone_non_aromatic_rings,
            backbone_aromatic_rings,
            backbone_topological_surface_area,
            backbone_fraction_bicyclic_rings,
            backbone_num_aliphatic_heterocycles,
            backbone_slogpvsa1,
            backbone_molecular_weight,
            backbone_sp3_carbon_count,
            backbone_sp2_carbon_count,
            backbone_max_estate_index,
            backbone_smr_vsa5,
            backbone_fp_density_morgan1,
            backbone_halogen_counts,
            backbone_bond_counts,
            backbone_bridging_rings_count,
            backbone_max_ring_size,
            backbone_heteroatom_density,
            backbone_heteroatom_count,
            sidechain_balabanjindex,
            sidechain_hbond_donors,
            sidechain_hbond_acceptors,
            sidechain_rotatable_bonds,
            sidechain_rings,
            sidechain_non_aromatic_rings,
            sidechain_aromatic_rings,
            sidechain_topological_surface_area,
            sidechain_fraction_bicyclic_rings,
            sidechain_num_aliphatic_heterocycles,
            sidechain_slogpvsa1,
            sidechain_molecular_weight,
            sidechain_sp3_carbon_count,
            sidechain_sp2_carbon_count,
            sidechain_max_estate_index,
            sidechain_smr_vsa5,
            sidechain_fp_density_morgan1,
            sidechain_halogen_counts,
            sidechain_bond_counts,
            sidechain_bridging_rings_count,
            sidechain_max_ring_size,
            sidechain_heteroatom_density,
            sidechain_heteroatom_count,
        ]
    )


def calculate_features(psmiles, featurizer):
    """Calculates features for a given polymer SMILES string using the specified featurizer.

    Args:
        psmiles (str): Polymer SMILES string to process.
        featurizer (MultipleFeaturizer): Configured featurizer to extract features.

    Returns:
        pd.Series: Series containing calculated features or None values if processing fails.
    """
    try:
        polymer_instance = Polymer.from_psmiles(psmiles)
        features = featurizer.featurize(polymer_instance)
        return pd.Series(features)
    except Exception as e:
        print(f"Error processing PSMILES {psmiles}: {str(e)}")
        return pd.Series([None] * len(featurizer.feature_labels()))


def process_csv(input_file, output_file, psmiles_column):
    """Processes a CSV file containing polymer SMILES and calculates their features.

    Args:
        input_file (str): Path to input CSV file.
        output_file (str): Path where output CSV file will be saved.
        psmiles_column (str): Name of the column containing PSMILES strings.
    """
    pandarallel.initialize(progress_bar=True)

    df = pd.read_csv(input_file)

    featurizer = create_featurizer()

    feature_df = df[psmiles_column].parallel_apply(
        lambda x: calculate_features(x, featurizer)
    )

    feature_df.columns = featurizer.feature_labels()

    result_df = pd.concat([df, feature_df], axis=1)

    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main(input_path, output_path, PSMILES_COLUMN="PSMILES"):
    """Main function to process polymer features from an input CSV file.

    Args:
        input_path (str): Path to input CSV file.
        output_path (str): Path where output CSV file will be saved.
        PSMILES_COLUMN (str, optional): Name of column containing PSMILES strings.
            Defaults to "PSMILES".
    """
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        print("Current working directory:", os.getcwd())
        print("Please make sure the input file exists and the input path is correct.")
        return

    process_csv(input_path, output_path, PSMILES_COLUMN)


if __name__ == "__main__":
    fire.Fire(main)
