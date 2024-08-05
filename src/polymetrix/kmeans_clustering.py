# import os
# from pathlib import Path
# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from polymetrix.core.utils import get_fp_ecfp_bitvector

# def kmeans_clustering(
#     data,
#     smiles_column,
#     fingerprint_column,
#     target_column,
#     cluster_range,
#     random_state,
#     init='k-means++',
#     max_iter=1000,
#     n_init=10
# ):
#     """Perform K-means clustering on molecular data.

#     Args:
#         data (pd.DataFrame): Input dataframe containing molecular data.
#         smiles_column (str): Name of the column containing SMILES strings.
#         fingerprint_column (str): Name of the column to store fingerprints.
#         target_column (str): Name of the column containing target values.
#         cluster_range (range): Range of cluster numbers to try.
#         random_state (int): Random state for reproducibility.
#         init (str, optional): Initialization method for K-means. Defaults to 'k-means++'.
#         max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
#         n_init (int, optional): Number of times to run k-means with different centroid seeds. Defaults to 10.

#     Returns:
#         tuple: A tuple containing:
#             - pd.DataFrame: Data with added cluster labels and source column.
#             - int: Optimal number of clusters.
#     """
#     data = data.drop_duplicates(subset=[smiles_column, target_column])
#     data[fingerprint_column] = data[smiles_column].apply(get_fp_ecfp_bitvector)
#     fingerprints_list = [list(fp) for fp in data[fingerprint_column]]

#     wcss = []
#     for num_clusters in cluster_range:
#         kmeans = KMeans(
#             n_clusters=num_clusters,
#             init=init,
#             max_iter=max_iter,
#             n_init=n_init,
#             random_state=random_state
#         )
#         kmeans.fit(fingerprints_list)
#         wcss.append(kmeans.inertia_)

#     optimal_k = find_optimal_k(cluster_range, wcss)

#     kmeans = KMeans(n_clusters=optimal_k, random_state=random_state)
#     data['cluster_label'] = kmeans.fit_predict(fingerprints_list)
#     data['Source'] = 'cluster_' + data['cluster_label'].astype(str)

#     return data, optimal_k

# def find_optimal_k(cluster_range, wcss):
#     """Find optimal K using the elbow method.

#     Args:
#         cluster_range (range): Range of cluster numbers tried.
#         wcss (list): List of within-cluster sum of squares for each K.

#     Returns:
#         int: Optimal number of clusters.
#     """
#     first_point = np.array([1, wcss[0]])
#     last_point = np.array([len(cluster_range), wcss[-1]])
#     line_vector = last_point - first_point

#     distances = []
#     for i, inertia in enumerate(wcss):
#         point = np.array([i + 1, inertia])
#         vector = point - first_point
#         distance = np.abs(np.cross(line_vector, vector)) / np.linalg.norm(line_vector)
#         distances.append(distance)

#     return distances.index(max(distances)) + 1

# def create_and_save_splits(data, output_dir, optimal_k, seed):
#     """Create and save train-test splits based on optimal clusters.

#     Args:
#         data (pd.DataFrame): Data with cluster labels.
#         output_dir (str): Directory to save output files.
#         optimal_k (int): Optimal number of clusters.
#         seed (int): Random seed used for clustering.
#     """
#     for test_cluster in range(optimal_k):
#         train_data = data[data['cluster_label'] != test_cluster]
#         test_data = data[data['cluster_label'] == test_cluster]

#         train_file = Path(output_dir) / f'train_data_extra_{seed}_cluster_{test_cluster}.csv'
#         test_file = Path(output_dir) / f'test_data_extra_{seed}_cluster_{test_cluster}.csv'

#         train_data.to_csv(train_file, index=False)
#         test_data.to_csv(test_file, index=False)

#         print(f"Saved train-test split for seed {seed}, test cluster {test_cluster}")

# def perform_clustering(
#     input_file,
#     output_dir,
#     random_seeds,
#     cluster_range,
#     smiles_column,
#     fingerprint_column,
#     target_column
# ):
#     """Perform clustering for multiple random seeds and save train-test splits.

#     Args:
#         input_file (str): Path to the input CSV file.
#         output_dir (str): Directory to save output files.
#         random_seeds (range): Range of random seeds to use.
#         cluster_range (range): Range of cluster numbers to try.
#         smiles_column (str): Name of the column containing SMILES strings.
#         fingerprint_column (str): Name of the column to store fingerprints.
#         target_column (str): Name of the column containing target values.
#     """
#     data = pd.read_csv(input_file)

#     Path(output_dir).mkdir(parents=True, exist_ok=True)

#     for seed in random_seeds:
#         clustered_data, optimal_k = kmeans_clustering(
#             data,
#             smiles_column,
#             fingerprint_column,
#             target_column,
#             cluster_range,
#             seed
#         )

#         output_file = Path(output_dir) / f'clustering_{seed}.csv'
#         clustered_data.to_csv(output_file, index=False)

#         print(f"Optimal K (number of clusters) for seed {seed}: {optimal_k}")

#         create_and_save_splits(clustered_data, output_dir, optimal_k, seed)

# if __name__ == "__main__":
#     perform_clustering(
#         input_file='/home/ta45woj/PolyMetriX/data/Polymer_Tg_descriptors.csv',
#         output_dir='/home/ta45woj/PolyMetriX/data_splits/extrapolation',
#         random_seeds=range(10, 11),
#         cluster_range=range(1, 10),
#         smiles_column='PSMILES',
#         fingerprint_column='fingerprint',
#         target_column='Exp_Tg(K)'
#     )

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from polymetrix.core.utils import get_fp_ecfp_bitvector

def kmeans_clustering(
    data,
    smiles_column,
    fingerprint_column,
    target_column,
    cluster_range,
    random_state,
    init='k-means++',
    max_iter=1000,
    n_init=10
):
    """Perform K-means clustering on molecular data.

    Args:
        data (pd.DataFrame): Input dataframe containing molecular data.
        smiles_column (str): Name of the column containing SMILES strings.
        fingerprint_column (str): Name of the column to store fingerprints.
        target_column (str): Name of the column containing target values.
        cluster_range (range): Range of cluster numbers to try.
        random_state (int): Random state for reproducibility.
        init (str, optional): Initialization method for K-means. Defaults to 'k-means++'.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        n_init (int, optional): Number of times to run k-means with different centroid seeds. Defaults to 10.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Data with added cluster labels and source column.
            - int: Optimal number of clusters.
    """
    data = data.drop_duplicates(subset=[smiles_column, target_column])
    data[fingerprint_column] = data[smiles_column].apply(get_fp_ecfp_bitvector)
    fingerprints_list = [list(fp) for fp in data[fingerprint_column]]

    wcss = []
    for num_clusters in cluster_range:
        kmeans = KMeans(
            n_clusters=num_clusters,
            init=init,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state
        )
        kmeans.fit(fingerprints_list)
        wcss.append(kmeans.inertia_)

    optimal_k = find_optimal_k(cluster_range, wcss)

    kmeans = KMeans(n_clusters=optimal_k, random_state=random_state)
    data['cluster_label'] = kmeans.fit_predict(fingerprints_list)
    data['Source'] = 'cluster_' + data['cluster_label'].astype(str)

    return data, optimal_k

def find_optimal_k(cluster_range, wcss):
    """Find optimal K using the elbow method.

    Args:
        cluster_range (range): Range of cluster numbers tried.
        wcss (list): List of within-cluster sum of squares for each K.

    Returns:
        int: Optimal number of clusters.
    """
    first_point = np.array([1, wcss[0]])
    last_point = np.array([len(cluster_range), wcss[-1]])
    line_vector = last_point - first_point

    distances = []
    for i, inertia in enumerate(wcss):
        point = np.array([i + 1, inertia])
        vector = point - first_point
        distance = np.abs(np.cross(line_vector, vector)) / np.linalg.norm(line_vector)
        distances.append(distance)

    return distances.index(max(distances)) + 1

def create_and_save_splits(data, output_dir, optimal_k, seed):
    """Create and save train-valid-test splits based on optimal clusters.

    Args:
        data (pd.DataFrame): Data with cluster labels.
        output_dir (str): Directory to save output files.
        optimal_k (int): Optimal number of clusters.
        seed (int): Random seed used for clustering.
    """
    for test_cluster in range(optimal_k):
        train_valid_data = data[data['cluster_label'] != test_cluster]
        test_data = data[data['cluster_label'] == test_cluster]

        # Split train_valid_data into train and validation sets
        train_data, valid_data = train_test_split(train_valid_data, test_size=0.1, random_state=seed)

        train_file = Path(output_dir) / f'train_data_extra_{seed}_cluster_{test_cluster}.csv'
        valid_file = Path(output_dir) / f'val_data_extra_{seed}_cluster_{test_cluster}.csv'
        test_file = Path(output_dir) / f'test_data_extra_{seed}_cluster_{test_cluster}.csv'

        train_data.to_csv(train_file, index=False)
        valid_data.to_csv(valid_file, index=False)
        test_data.to_csv(test_file, index=False)

        print(f"Saved train-valid-test split for seed {seed}, test cluster {test_cluster}")

def perform_clustering(
    input_file,
    output_dir,
    random_seeds,
    cluster_range,
    smiles_column,
    fingerprint_column,
    target_column
):
    """Perform clustering for multiple random seeds and save train-test splits.

    Args:
        input_file (str): Path to the input CSV file.
        output_dir (str): Directory to save output files.
        random_seeds (range): Range of random seeds to use.
        cluster_range (range): Range of cluster numbers to try.
        smiles_column (str): Name of the column containing SMILES strings.
        fingerprint_column (str): Name of the column to store fingerprints.
        target_column (str): Name of the column containing target values.
    """
    data = pd.read_csv(input_file)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for seed in random_seeds:
        clustered_data, optimal_k = kmeans_clustering(
            data,
            smiles_column,
            fingerprint_column,
            target_column,
            cluster_range,
            seed
        )

        output_file = Path(output_dir) / f'clustering_{seed}.csv'
        clustered_data.to_csv(output_file, index=False)

        print(f"Optimal K (number of clusters) for seed {seed}: {optimal_k}")

        create_and_save_splits(clustered_data, output_dir, optimal_k, seed)

if __name__ == "__main__":
    perform_clustering(
        input_file='/home/ta45woj/PolyMetriX/data/Polymer_Tg_descriptors.csv',
        output_dir='/home/ta45woj/PolyMetriX/data_splits/extrapolation',
        random_seeds=range(10, 11),
        cluster_range=range(1, 10),
        smiles_column='PSMILES',
        fingerprint_column='fingerprint',
        target_column='Exp_Tg(K)'
    )