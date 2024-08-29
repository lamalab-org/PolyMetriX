import pandas as pd
from pandarallel import pandarallel
from polymetrix.polymer import Polymer
from polymetrix.featurizer import (
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
    SideChainFeaturizer,
    BackBoneFeaturizer,
    FullPolymerFeaturizer,
    NumSideChainFeaturizer,
    NumBackBoneFeaturizer,
    MultipleFeaturizer,
    MolecularWeightFeaturizer,
    Sp3CarbonCountFeaturizer,
    Sp2CarbonCountFeaturizer,
)
import os

# Configuration
DATA_DIR = "/home/ta45woj/PolyMetriX/data"
INPUT_FILE = "Polymer_Tg.csv"
OUTPUT_FILE = "Polymer_Tg_featurizers.csv"
PSMILES_COLUMN = "PSMILES"


def create_featurizer():
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
        ]
    )


def calculate_features(psmiles, featurizer):
    try:
        polymer_instance = Polymer.from_psmiles(psmiles)
        features = featurizer.featurize(polymer_instance)
        return pd.Series(features)
    except Exception as e:
        print(f"Error processing PSMILES {psmiles}: {str(e)}")
        return pd.Series([None] * len(featurizer.feature_labels()))


def process_csv(input_file, output_file, psmiles_column):
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


def main():
    input_path = os.path.join(DATA_DIR, INPUT_FILE)
    output_path = os.path.join(DATA_DIR, OUTPUT_FILE)

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        print("Current working directory:", os.getcwd())
        print(
            "Please make sure the input file exists and the DATA_DIR path is correct."
        )
        return

    process_csv(input_path, output_path, PSMILES_COLUMN)


if __name__ == "__main__":
    main()