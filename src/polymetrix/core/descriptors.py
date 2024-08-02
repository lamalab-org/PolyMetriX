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
        mol (rdkit.Chem.rdchem.Mol): An RDKit molecule object.

    Returns:
        networkx.Graph: A NetworkX graph representation of the molecule.
    """
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), element=atom.GetSymbol())
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        order = bond.GetBondType()
        graph.add_edge(start, end, order=order)

    return graph


def find_shortest_paths_between_stars(graph):
    """Find the shortest paths between star nodes in a graph.

    Args:
        graph (networkx.Graph): A NetworkX graph.

    Returns:
        list: A list of shortest paths between star nodes.
    """
    star_nodes = [node for node, data in graph.nodes(data=True)
                  if data['element'] == '*']
    shortest_paths = []
    for i in range(len(star_nodes)):
        for j in range(i + 1, len(star_nodes)):
            try:
                path = nx.shortest_path(graph, source=star_nodes[i],
                                        target=star_nodes[j])
                shortest_paths.append(path)
            except nx.NetworkXNoPath:
                continue

    return shortest_paths


def find_cycles_including_paths(graph, paths):
    """Find cycles in a graph that include given paths.

    Args:
        graph (networkx.Graph): A NetworkX graph.
        paths (list): A list of paths.

    Returns:
        list: A list of cycles that include the given paths.
    """
    cycles = set()
    for path in paths:
        for node in path:
            try:
                all_cycles = nx.cycle_basis(graph, node)
                for cycle in all_cycles:
                    if any(n in path for n in cycle):
                        sorted_cycle = tuple(sorted((min(c), max(c))
                                             for c in zip(cycle, cycle[1:] + [cycle[0]])))
                        cycles.add(sorted_cycle)
            except nx.NetworkXNoCycle:
                continue

    return [list(cycle) for cycle in cycles]


def add_degree_one_nodes_to_backbone(graph, backbone):
    """Add degree one nodes to the backbone of a graph.

    Args:
        graph (networkx.Graph): A NetworkX graph.
        backbone (list): A list of backbone nodes.

    Returns:
        list: An updated list of backbone nodes including degree one nodes.
    """
    for node in list(graph.nodes):
        if graph.degree[node] == 1:
            neighbor = next(iter(graph.neighbors(node)))
            if neighbor in backbone:
                backbone.append(node)

    return backbone


def classify_backbone_and_sidechains(graph):
    """Classify nodes in a graph as backbone or sidechain nodes.

    Args:
        graph (networkx.Graph): A NetworkX graph.

    Returns:
        tuple: A tuple containing lists of backbone and sidechain nodes.
    """
    shortest_paths = find_shortest_paths_between_stars(graph)
    cycles = find_cycles_including_paths(graph, shortest_paths)
    backbone_nodes = set()
    for cycle in cycles:
        for edge in cycle:
            backbone_nodes.update(edge)
    for path in shortest_paths:
        backbone_nodes.update(path)
    backbone_nodes = add_degree_one_nodes_to_backbone(graph, list(backbone_nodes))
    sidechain_nodes = [node for node in graph.nodes if node not in backbone_nodes]

    return list(set(backbone_nodes)), sidechain_nodes


def get_real_backbone_and_sidechain_bridges(graph, backbone_nodes, sidechain_nodes):
    """Get the real backbone and sidechain bridges in a graph.

    Args:
        graph (networkx.Graph): A NetworkX graph.
        backbone_nodes (list): A list of backbone nodes.
        sidechain_nodes (list): A list of sidechain nodes.

    Returns:
        tuple: A tuple containing lists of backbone and sidechain bridges.
    """
    backbone_bridges = []
    sidechain_bridges = []
    for edge in graph.edges:
        start_node, end_node = edge
        if start_node in backbone_nodes and end_node in backbone_nodes:
            backbone_bridges.append(edge)
        elif start_node in sidechain_nodes or end_node in sidechain_nodes:
            sidechain_bridges.append(edge)

    return backbone_bridges, sidechain_bridges


def number_and_length_of_sidechains_and_backbones(sidechain_bridges, backbone_bridges):
    """Merge common bridges from the side chain bridges and backbone bridges.

    Args:
        sidechain_bridges (list): A list of sidechain bridges.
        backbone_bridges (list): A list of backbone bridges.

    Returns:
        tuple: A tuple containing lists of sidechains and backbones.
    """
    sidechain_graph = nx.Graph(sidechain_bridges)
    backbone_graph = nx.Graph(backbone_bridges)
    sidechains = list(nx.connected_components(sidechain_graph))
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