import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from collections import OrderedDict
from pandarallel import pandarallel
import networkx as nx
from radonpy.core.poly import make_linearpolymer


def mol_from_smiles(psmiles):
    """
    Convert a PSMILES string to an RDKit molecule object.

    Args:
        psmiles (str): PSMILES string.

    Returns:
        rdkit.Chem.Mol: RDKit molecule object.
    """
    return Chem.MolFromSmiles(psmiles)


def polymer_from_smiles(psmiles, degree=2):
    """
    Generate a linear polymer from a PSMILES string.
    
    Args:
        psmiles (str): SMILES string of the monomer.
        degree (int): Degree of the polymer.
    
    Returns:
        str: PSMILES string of the polymer.
    """
    deg_smiles = make_linearpolymer(psmiles, degree)
    
    return deg_smiles


def generate_conformers(mol, 
                        num_confs=500, 
                        seed=100, 
                        max_iters=1000, 
                        num_threads=5, 
                        prune_rms_thresh=0.5, 
                        non_bonded_thresh=100.0):
    """
    Generate conformers for a given molecule.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object.
        num_confs (int): Number of conformers to generate.
        max_iters (int): Maximum number of iterations for conformer generation.
        num_threads (int): Number of threads to use.
        prune_rms_thresh (float): RMS threshold for pruning conformers.
        non_bonded_thresh (float): Non-bonded threshold for conformer generation.
        seed (int): Random seed for reproducibility.

    Returns:
        list: List of energies of the generated conformers.
    """
    params = AllChem.ETKDGv3()
    params.useSmallRingTorsions = True

    molecule = Chem.AddHs(mol)
    conformers = AllChem.EmbedMultipleConfs(
        molecule, numConfs=num_confs, randomSeed=seed, pruneRmsThresh=prune_rms_thresh, numThreads=num_threads
    )
    print(f"Generated {len(conformers)} conformers")

    try:
        optimised_and_energies = AllChem.MMFFOptimizeMoleculeConfs(
            molecule, maxIters=max_iters, numThreads=num_threads, nonBondedThresh=non_bonded_thresh
        )
    except Exception as e:
        print(f"Optimization failed: {e}")
        return []

    energy_dict = {conf: energy for conf, (optimized, energy) in zip(conformers, optimised_and_energies) if optimized == 0}
    if not energy_dict:
        return []

    molecule = AllChem.RemoveHs(molecule)
    matches = molecule.GetSubstructMatches(molecule, uniquify=False)
    maps = [list(enumerate(match)) for match in matches]
    final_conformers = OrderedDict()

    for conf_id, energy in sorted(energy_dict.items(), key=lambda x: x[1]):
        if all(AllChem.GetBestRMS(molecule, molecule, ref_id, conf_id, maps) >= 1.0 for ref_id in final_conformers):
            final_conformers[conf_id] = energy

    return list(final_conformers.values())


def calc_nconf20(energy_list):
    """
    Calculate the number of conformers within 20 kcal/mol of the lowest energy conformer.

    Args:
        energy_list (list): List of conformer energies.

    Returns:
        int: Number of conformers within 20 kcal/mol of the lowest energy conformer.
    """
    if not energy_list:
        return 1
    energy_array = np.array(energy_list)
    relative_energies = energy_array - energy_array[0]
    return np.sum((0 <= relative_energies) & (relative_energies < 20))


def n_conf20(psmiles, num_confs=500, seed=100):
    """
    Calculate the n_conf20 descriptor for a given SMILES string.

    Args:
        psmiles (str): PSMILES string of the molecule with degree 2.
        seed (int): Random seed for reproducibility.

    Returns:
        float: n_conf20 descriptor value.
    """
    try:
        mol = Chem.MolFromSmiles(psmiles)
        if mol is None:
            return np.nan
        energy_list = generate_conformers(mol, num_confs=num_confs, seed=seed)
        return calc_nconf20(energy_list)
    except Exception as e:
        print(f"Failed to compute descriptor for {psmiles}: {e}")
        return np.nan


def calculate_hbond_acceptors(mol):
    """
    Calculate the number of hydrogen bond acceptors for a given RDKit molecule object.
    
    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object.
    
    Returns:
        int: Number of hydrogen bond acceptors.
    """
    return Descriptors.NumHAcceptors(mol)


def calculate_hbond_donors(mol):
    """
    Calculate the number of hydrogen bond donors for a given RDKit molecule object.
    
    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object.
    
    Returns:
        int: Number of hydrogen bond donors.
    """
    return Descriptors.NumHDonors(mol)


def calculate_rotatable_bonds(mol):
    """
    Calculate the number of rotatable bonds for a given RDKit molecule object.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object.

    Returns:
        int: Number of rotatable bonds.
    """
    return Descriptors.NumRotatableBonds(mol)


def calculate_ring_info(mol):
    """
    Calculate ring information for a given RDKit molecule object.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object.

    Returns:
        tuple: Number of rings, number of aromatic rings, number of non-aromatic rings.
    """
    aromatic_rings = [
        ring for ring in Chem.GetSymmSSSR(mol) if mol.GetRingInfo().IsAtomInRingOfSize(ring[0], len(ring)) and mol.GetAtomWithIdx(ring[0]).GetIsAromatic()
    ]
    non_aromatic_rings = [
        ring for ring in Chem.GetSymmSSSR(mol) if mol.GetRingInfo().IsAtomInRingOfSize(ring[0], len(ring)) and not mol.GetAtomWithIdx(ring[0]).GetIsAromatic()
    ]
    return mol.GetRingInfo().NumRings(), len(aromatic_rings), len(non_aromatic_rings)


def mol_to_nx(mol):
    """Convert an RDKit molecule to a NetworkX graph.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object.

    Returns:
        networkx.Graph: NetworkX graph representation of the molecule.
    """
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), element=atom.GetSymbol())

    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        order = bond.GetBondType()
        G.add_edge(start, end, order=order)

    return G


def find_shortest_paths_between_stars(G):
    """Find the shortest paths between star nodes in a graph.

    Args:
        G (networkx.Graph): NetworkX graph.

    Returns:
        list: List of shortest paths between star nodes.
    """
    star_nodes = [node for node, data in G.nodes(data=True) if data['element'] == '*']
    shortest_paths = []
    for i in range(len(star_nodes)):
        for j in range(i + 1, len(star_nodes)):
            try:
                path = nx.shortest_path(G, source=star_nodes[i], target=star_nodes[j])
                shortest_paths.append(path)
            except nx.NetworkXNoPath:
                continue
    return shortest_paths


def find_cycles_including_paths(G, paths):
    """Find cycles in a graph that include given paths.

    Args:
        G (networkx.Graph): NetworkX graph.
        paths (list): List of paths.

    Returns:
        list: List of cycles that include the given paths.
    """
    cycles = set()
    for path in paths:
        for node in path:
            try:
                all_cycles = nx.cycle_basis(G, node)
                for cycle in all_cycles:
                    if any(n in path for n in cycle):
                        cycles.add(
                            tuple(
                                sorted(
                                    (min(cycle), max(cycle)) for cycle in zip(cycle, cycle[1:] + [cycle[0]])
                                )
                            )
                        )
            except nx.NetworkXNoCycle:
                continue
    return [list(cycle) for cycle in cycles]


def add_degree_one_nodes_to_backbone(G, backbone):
    """Add degree one nodes to the backbone of a graph.

    Args:
        G (networkx.Graph): NetworkX graph.
        backbone (list): List of backbone nodes.

    Returns:
        list: Updated list of backbone nodes.
    """
    for node in list(G.nodes):
        if G.degree[node] == 1:
            neighbor = list(G.neighbors(node))[0]
            if neighbor in backbone:
                backbone.append(node)
    return backbone


def classify_backbone_and_sidechains(G):
    """Classify nodes in a graph as backbone or sidechain nodes.

    Args:
        G (networkx.Graph): NetworkX graph.

    Returns:
        tuple: List of backbone nodes, list of sidechain nodes.
    """
    shortest_paths = find_shortest_paths_between_stars(G)
    cycles = find_cycles_including_paths(G, shortest_paths)

    backbone_nodes = set()
    for cycle in cycles:
        for edge in cycle:
            backbone_nodes.update(edge)
    for path in shortest_paths:
        backbone_nodes.update(path)

    backbone_nodes = add_degree_one_nodes_to_backbone(G, list(backbone_nodes))
    sidechain_nodes = [node for node in G.nodes if node not in backbone_nodes]

    return list(set(backbone_nodes)), sidechain_nodes


def get_real_backbone_and_sidechain_bridges(G, backbone_nodes, sidechain_nodes):
    """
    Get the real backbone and sidechain bridges in a graph.

    Parameters:
    G (networkx.Graph): NetworkX graph.
    backbone_nodes (list): List of backbone nodes.
    sidechain_nodes (list): List of sidechain nodes.

    Returns:
    tuple: List of backbone bridges, list of sidechain bridges.
    """
    backbone_bridges = []
    sidechain_bridges = []

    for edge in G.edges:
        start_node, end_node = edge
        if start_node in backbone_nodes and end_node in backbone_nodes:
            backbone_bridges.append(edge)
        elif start_node in sidechain_nodes or end_node in sidechain_nodes:
            sidechain_bridges.append(edge)

    return backbone_bridges, sidechain_bridges


def group_side_chain_and_backbone_bridges(side_chain_bridges, backbone_bridges):
    """
    Groups the side chain bridges and backbone bridges into separate tuples of tuples,
    where each inner tuple represents a continuous side chain or backbone.
    Also returns the lengths of these side chains and backbones as separate lists.

    Args:
        side_chain_bridges (list): A list of tuples representing the side chain bridges.
        backbone_bridges (list): A list of tuples representing the backbone bridges.

    Returns:
        tuple: A tuple of tuples, where each inner tuple represents a continuous side chain.
        list: A list containing the lengths of the side chains.
        tuple: A tuple of tuples, where each inner tuple represents a continuous backbone.
        list: A list containing the lengths of the backbones.
    """
    if not side_chain_bridges:
        side_chain_bridges = ()
        side_chain_lengths = []
    else:
        grouped_side_chains = []
        side_chain_lengths = []
        current_side_chain = []

        for bridge in side_chain_bridges:
            if not current_side_chain or bridge[0] in current_side_chain[-1] or bridge[1] in current_side_chain[-1]:
                current_side_chain.append(bridge)
            else:
                grouped_side_chains.append(tuple(current_side_chain))
                side_chain_lengths.append(len(current_side_chain))
                current_side_chain = [bridge]

        if current_side_chain:
            grouped_side_chains.append(tuple(current_side_chain))
            side_chain_lengths.append(len(current_side_chain))

        side_chain_bridges = tuple(grouped_side_chains)

    if not backbone_bridges:
        backbone_bridges = ()
        backbone_lengths = []
    else:
        grouped_backbones = []
        backbone_lengths = []
        current_backbone = []

        for bridge in backbone_bridges:
            if not current_backbone or bridge[0] in current_backbone[-1] or bridge[1] in current_backbone[-1]:
                current_backbone.append(bridge)
            else:
                grouped_backbones.append(tuple(current_backbone))
                backbone_lengths.append(len(current_backbone))
                current_backbone = [bridge]

        if current_backbone:
            grouped_backbones.append(tuple(current_backbone))
            backbone_lengths.append(len(current_backbone))

        backbone_bridges = tuple(grouped_backbones)

    return side_chain_bridges, side_chain_lengths, backbone_bridges, backbone_lengths


def number_and_length_of_sidechains_and_backbones(grouped_side_chains, grouped_backbones):
    """
    Merges common bridges from the grouped side chain bridges and backbone bridges.

    Args:
        grouped_side_chains (tuple): A tuple of tuples, where each inner tuple represents a continuous side chain.
        grouped_backbones (tuple): A tuple of tuples, where each inner tuple represents a continuous backbone.

    Returns:
        list: A list of sets, where each set represents a sidechain.
        list: A list of sets, where each set represents a backbone.
    """

    all_side_chain_groups = [bridge for group in grouped_side_chains for bridge in group]
    side_chain_graph = nx.Graph(all_side_chain_groups)
    sidechains = list(nx.connected_components(side_chain_graph))

    all_backbone_groups = [bridge for group in grouped_backbones for bridge in group]
    backbone_graph = nx.Graph(all_backbone_groups)
    backbones = list(nx.connected_components(backbone_graph))

    return sidechains, backbones


def process_and_save(df, PSMILES_deg_col, output_file, batch_size=1):
    """Process the DataFrame in batches and save the results periodically.

    Args:
        df (pd.DataFrame): DataFrame to process.
        PSMILES_deg_col (str): Column name of the PSMILES with degree 2.
        output_file (str): Path to the output CSV file.
        batch_size (int): Number of rows to process in each batch.

    """
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch = df.iloc[start:end]
        batch['nconf20_2'] = batch[PSMILES_deg_col].parallel_apply(n_conf20)
        df.update(batch)
        df.to_csv(output_file, index=False)
        print(f"Processed rows {start} to {end} and saved to {output_file}")


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
    df['hbond_acceptors'] = df[PSMILES_deg_col].apply(lambda smiles: calculate_hbond_acceptors(mol_from_smiles(smiles)))
    df['hbond_donors'] = df[PSMILES_deg_col].apply(lambda smiles: calculate_hbond_donors(mol_from_smiles(smiles)))
    df['rotatable_bonds'] = df[PSMILES_deg_col].apply(lambda smiles: calculate_rotatable_bonds(mol_from_smiles(smiles)))
    df['num_rings'], df['num_aromatic_rings'], df['num_non_aromatic_rings'] = zip(
        *df[PSMILES_deg_col].apply(lambda smiles: calculate_ring_info(mol_from_smiles(smiles)))
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
            df.at[index, 'n_sc'] = len(sidechains)
            df.at[index, 'n_bb'] = len(backbones)

            if sidechains:
                sidechain_lengths = [len(sidechain) for sidechain in sidechains]
                df.at[index, 'len_sc'] = str(sidechain_lengths)
                df.at[index, 'min_length_sc'] = min(sidechain_lengths)
                df.at[index, 'max_length_sc'] = max(sidechain_lengths)
                df.at[index, 'mean_length_sc'] = sum(sidechain_lengths) / len(sidechain_lengths)
            else:
                df.at[index, 'len_sc'] = ''
                df.at[index, 'min_length_sc'] = 0
                df.at[index, 'max_length_sc'] = 0
                df.at[index, 'mean_length_sc'] = 0.0

            if backbones:
                backbone_lengths = [len(backbone) for backbone in backbones]
                df.at[index, 'len_bb'] = str(backbone_lengths)
                df.at[index, 'min_length_bb'] = min(backbone_lengths)
                df.at[index, 'max_length_bb'] = max(backbone_lengths)
                df.at[index, 'mean_length_bb'] = sum(backbone_lengths) / len(backbone_lengths)
            else:
                df.at[index, 'len_bb'] = ''
                df.at[index, 'min_length_bb'] = 0
                df.at[index, 'max_length_bb'] = 0
                df.at[index, 'mean_length_bb'] = 0.0
        except ValueError:
            pass

    # Save the final DataFrame to the output file
    df.to_csv(output_file, index=False)
    print(f"Final DataFrame saved to {output_file}")


if __name__ == "__main__":
    input_file = os.path.join(os.path.expanduser("~"), "Poly_descriptors", "data", "Polymer_Tg.csv")
    output_file = os.path.join(os.path.expanduser("~"), "Poly_descriptors", "data", "Polymer_Tg_descriptors.csv")
    PSMILES_deg_col = 'dp_2'
    PSMILES = 'PSMILES'
    main(input_file, PSMILES_deg_col, PSMILES, output_file)