import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from collections import OrderedDict
from pandarallel import pandarallel
import networkx as nx

pandarallel.initialize(progress_bar=True)

def mol_from_smiles(smiles):
    """
    Convert a SMILES string to an RDKit molecule object.

    Parameters:
    smiles (str): SMILES string.

    Returns:
    rdkit.Chem.Mol: RDKit molecule object.
    """
    return Chem.MolFromSmiles(smiles)


def generate_conformers(mol, 
                        num_confs=500, 
                        seed=100, 
                        max_iters=1000, 
                        num_threads=5, 
                        prune_rms_thresh=0.5, 
                        non_bonded_thresh=100.0
                        ):
    """
    Generate conformers for a given molecule.

    Parameters:
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

    Parameters:
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

    Parameters:
    smiles (str): PSMILES string of the molecule with degree 2.
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


def calculate_rotatable_bonds(mol):
    """
    Calculate the number of rotatable bonds for a given RDKit molecule object.

    Parameters:
    mol (rdkit.Chem.Mol): RDKit molecule object.

    Returns:
    int: Number of rotatable bonds.
    """
    return Descriptors.NumRotatableBonds(mol)


def calculate_ring_info(mol):
    """
    Calculate ring information for a given RDKit molecule object.

    Parameters:
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
    """
    Convert an RDKit molecule to a NetworkX graph.

    Parameters:
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
    """
    Find the shortest paths between star nodes in a graph.

    Parameters:
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
    """
    Find cycles in a graph that include given paths.

    Parameters:
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
    """
    Add degree one nodes to the backbone of a graph.

    Parameters:
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
    """
    Classify nodes in a graph as backbone or sidechain nodes.

    Parameters:
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

    return list(backbone_nodes), sidechain_nodes


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


def group_side_chain_bridges(side_chain_bridges):
    """
    Group sidechain bridges in a graph.

    Parameters:
    side_chain_bridges (list): List of sidechain bridges.

    Returns:
    tuple: Grouped sidechains, list of sidechain lengths.
    """
    if not side_chain_bridges:
        return (), []

    grouped_side_chains = []
    side_chain_lengths = []
    current_side_chain = []

    for i in range(len(side_chain_bridges)):
        bridge = side_chain_bridges[i]
        if not current_side_chain or bridge[0] in current_side_chain[-1]:
            current_side_chain.append(bridge)
        else:
            grouped_side_chains.append(tuple(current_side_chain))
            side_chain_lengths.append(len(current_side_chain))
            current_side_chain = [bridge]

    if current_side_chain:
        grouped_side_chains.append(tuple(current_side_chain))
        side_chain_lengths.append(len(current_side_chain))

    return tuple(grouped_side_chains), side_chain_lengths


def number_and_length_of_sidechains(grouped_side_chains):
    """
    Calculate the number and length of sidechains in a graph.

    Parameters:
    grouped_side_chains (tuple): Grouped sidechains.

    Returns:
    list: List of sidechains.
    """
    def sort_tuple(t):
        return sorted(t)

    sorted_groups = list(map(sort_tuple, grouped_side_chains))
    all_groups = []
    for g in sorted_groups:
        all_groups.extend(g)

    G = nx.Graph(all_groups)
    sidechains = list(nx.connected_components(G))

    return sidechains


def process_and_save(df, output_file, batch_size=1):
    """
    Process the DataFrame in batches and save the results periodically.

    Parameters:
    df (pd.DataFrame): DataFrame to process.
    output_file (str): Path to the output CSV file.
    batch_size (int): Number of rows to process in each batch.
    """
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch = df.iloc[start:end]
        batch['nconf20_2'] = batch['dp_2'].parallel_apply(n_conf20)
        df.update(batch)
        df.to_csv(output_file, index=False)
        print(f"Processed rows {start} to {end} and saved to {output_file}")


def main(input_file, output_file):
    """
    Main function to process the input CSV file and save the results.

    Parameters:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to the output CSV file.
    """
    # Check if the output file already exists
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
    else:
        df = pd.read_csv(input_file)
        print(f"Initial DataFrame shape: {df.shape}")
        df['nconf20_2'] = np.nan

    # Process each row in parallel using pandarallel and save periodically
    process_and_save(df, output_file)

    # Calculate 2D descriptors
    df['rotatable_bonds'] = df['dp_2'].apply(lambda smiles: calculate_rotatable_bonds(mol_from_smiles(smiles)))
    df['num_rings'], df['num_aromatic_rings'], df['num_non_aromatic_rings'] = zip(
        *df['dp_2'].apply(lambda smiles: calculate_ring_info(mol_from_smiles(smiles)))
    )

    # Initialize new columns for sidechain information
    df['n_sc'] = 0
    df['len_sc'] = ''
    df['min_length'] = 0
    df['max_length'] = 0
    df['mean_length'] = 0.0

    # Iterate over each PSMILES in the DataFrame
    for index, row in df.iterrows():
        psmiles = row['PSMILES']
        mol = Chem.MolFromSmiles(psmiles)
        if mol is None:
            continue

        G = mol_to_nx(mol)
        backbone_nodes, sidechain_nodes = classify_backbone_and_sidechains(G)
        backbone_bridges, sidechain_bridges = get_real_backbone_and_sidechain_bridges(G, backbone_nodes, sidechain_nodes)

        try:
            grouped_side_chains, _ = group_side_chain_bridges(sidechain_bridges)
            sidechains = number_and_length_of_sidechains(grouped_side_chains)
            df.at[index, 'n_sc'] = len(sidechains)
            sidechain_lengths = [len(sidechain) for sidechain in sidechains]
            df.at[index, 'len_sc'] = str(sidechain_lengths)
            df.at[index, 'min_length'] = min(sidechain_lengths)
            df.at[index, 'max_length'] = max(sidechain_lengths)
            df.at[index, 'mean_length'] = sum(sidechain_lengths) / len(sidechain_lengths)
        except ValueError:
            pass

    # Save the final DataFrame to the output file
    df.to_csv(output_file, index=False)
    print(f"Final DataFrame saved to {output_file}")


if __name__ == "__main__":
    input_file = os.path.join(os.path.expanduser("~"), "data", "Polymer_Tg.csv")
    output_file = os.path.join(os.path.expanduser("~"), "data", "Polymer_Tg_descriptors.csv")
    main(input_file, output_file)
    
