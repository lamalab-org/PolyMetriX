import os
import pandas as pd
import numpy as np 
from pandarallel import pandarallel 
from pathlib import Path 
from polymetrix.core.descriptors import (
    calculate_hbond_acceptors,
    calculate_hbond_donors,
    calculate_ring_info,
    calculate_rotatable_bonds,
    classify_backbone_and_sidechains,
    get_real_backbone_and_sidechain_bridges,
    group_side_chain_and_backbone_bridges,
    mol_from_smiles,
    mol_to_nx,
    number_and_length_of_sidechains_and_backbones,
    polymer_from_smiles,
    process_and_save,
)


def main(input_file, PSMILES_deg_col, PSMILES, output_file):
    """Main function to process the input CSV file and save the results.

    Args:
        input_file (str): Path to the input CSV file.
        PSMILES_deg_col (str): Column name of the PSMILES with degree 2.
        PSMILES (str): Column name of the PSMILES.
        output_file (str): Path to the output CSV file.

    """
    # Initialize pandarallel
    pandarallel.initialize(progress_bar=True)

    try:
        df = pd.read_csv(input_file)
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return

    # Create the dp_2 column without saving to input file
    df[PSMILES_deg_col] = df[PSMILES].apply(lambda x: polymer_from_smiles(x))

    # Check if the output file already exists
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
    else:
        df = df.copy()
        df['nconf20_2'] = np.nan

    # Ensure dp_2 is created and populated before processing
    df[PSMILES_deg_col] = df[PSMILES].apply(lambda x: polymer_from_smiles(x))

    # Process each row in parallel using pandarallel and save periodically
    process_and_save(df, PSMILES_deg_col, output_file)

    # Calculate 2D descriptors
    df['hbond_acceptors'] = df[PSMILES_deg_col].apply(
        lambda smiles: calculate_hbond_acceptors(mol_from_smiles(smiles))
    )
    df['hbond_donors'] = df[PSMILES_deg_col].apply(
        lambda smiles: calculate_hbond_donors(mol_from_smiles(smiles))
    )
    df['rotatable_bonds'] = df[PSMILES_deg_col].apply(
        lambda smiles: calculate_rotatable_bonds(mol_from_smiles(smiles))
    )
    df['num_rings'], df['num_aromatic_rings'], df['num_non_aromatic_rings'] = zip(*df[PSMILES_deg_col].apply(
        lambda smiles: calculate_ring_info(mol_from_smiles(smiles)))
    )

    # Initialize new columns for sidechain and backbone information
    df['n_sc'] = 0
    df['len_sc'] = ''
    df['min_length_sc'] = 0
    df['max_length_sc'] = 0
    df['mean_length_sc'] = 0.0
    df['n_bb'] = 0
    df['len_bb'] = ''
    df['min_length_bb'] = 0
    df['max_length_bb'] = 0
    df['mean_length_bb'] = 0.0

    # Iterate over each PSMILES in the DataFrame
    for index, row in df.iterrows():
        psmiles = row[PSMILES]
        mol = mol_from_smiles(psmiles)
        if mol is None:
            continue

        G = mol_to_nx(mol)
        backbone_nodes, sidechain_nodes = classify_backbone_and_sidechains(G)
        backbone_bridges, sidechain_bridges = get_real_backbone_and_sidechain_bridges(G, backbone_nodes, sidechain_nodes)

        try:
            side_chain_bridges, side_chain_lengths, backbone_bridges, backbone_lengths = group_side_chain_and_backbone_bridges(sidechain_bridges, backbone_bridges)
            sidechains, backbones = number_and_length_of_sidechains_and_backbones(side_chain_bridges, backbone_bridges)
            df.loc[index, 'n_sc'] = len(sidechains)
            df.loc[index, 'n_bb'] = len(backbones)

            if sidechains:
                sidechain_lengths = [len(sidechain) for sidechain in sidechains]
                df.loc[index, 'len_sc'] = str(sidechain_lengths)
                df.loc[index, 'min_length_sc'] = min(sidechain_lengths)
                df.loc[index, 'max_length_sc'] = max(sidechain_lengths)
                df.loc[index, 'mean_length_sc'] = sum(sidechain_lengths) / len(sidechain_lengths)
            else:
                df.loc[index, 'len_sc'] = ''
                df.loc[index, 'min_length_sc'] = 0
                df.loc[index, 'max_length_sc'] = 0
                df.loc[index, 'mean_length_sc'] = 0.0

            if backbones:
                backbone_lengths = [len(backbone) for backbone in backbones]
                df.loc[index, 'len_bb'] = str(backbone_lengths)
                df.loc[index, 'min_length_bb'] = min(backbone_lengths)
                df.loc[index, 'max_length_bb'] = max(backbone_lengths)
                df.loc[index, 'mean_length_bb'] = sum(backbone_lengths) / len(backbone_lengths)
            else:
                df.loc[index, 'len_bb'] = ''
                df.loc[index, 'min_length_bb'] = 0
                df.loc[index, 'max_length_bb'] = 0
                df.loc[index, 'mean_length_bb'] = 0.0
        except ValueError:
            pass

    # Save the final DataFrame to the output file
    df.to_csv(output_file, index=False)
    print(f"Final DataFrame saved to {output_file}")


if __name__ == "__main__":
    input_file = Path.home().joinpath("Poly_descriptors", "data", "dummy.csv")
    output_file = Path.home().joinpath("Poly_descriptors", "data", "Polymer_Tg_descriptors_dummy.csv")
    PSMILES_deg_col = 'dp_2'
    PSMILES = 'PSMILES'
    main(input_file, PSMILES_deg_col, PSMILES, output_file)