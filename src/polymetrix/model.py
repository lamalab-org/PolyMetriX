import json
from pathlib import Path
import statistics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
import optuna
from polymetrix.core.utils import get_fp_ecfp_bitvector, calculate_metrics
import hydra
from omegaconf import DictConfig


def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)


def interpolation_split_and_save_data(data, cfg):
    """
    Split data into train, validation and test sets and save them in an interpolation folder.

    Args:
        data (pd.DataFrame): Input data.
        cfg (DictConfig): Configuration object containing parameters.

    Returns:
        tuple: Train, validation and test data.
    """
    train_data, test_data = train_test_split(
        data,
        test_size=cfg.test_size,
        random_state=cfg.random_state
    )
    train_data, val_data = train_test_split(
        train_data,
        test_size=cfg.val_size,
        random_state=cfg.random_state
    )

    output_dir = Path(cfg.output_dir)
    interpolation_dir = output_dir / 'interpolation'
    interpolation_dir.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(interpolation_dir / 'train_data.csv', index=False)
    val_data.to_csv(interpolation_dir / 'val_data.csv', index=False)
    test_data.to_csv(interpolation_dir / 'test_data.csv', index=False)

    return train_data, val_data, test_data


def get_features(df, cfg, feature_type='all', features=None, fp_bits=2048):
    """
    Extract features from the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        feature_type (str): Type of features to extract.
        features (list): List of feature names.
        fp_bits (int): Number of bits for fingerprint.

    Returns:
        np.ndarray: Extracted features.
    """
    if feature_type in ['fingerprints', 'all']:
        fingerprints = df[cfg['smiles_col']].apply(
            lambda x: get_fp_ecfp_bitvector(x, nBits=fp_bits)
        )
        fp_array = np.array([
            list(fp) if fp is not None else [0] * fp_bits for fp in fingerprints
        ])

    if feature_type == 'fingerprints':
        return fp_array
    elif feature_type == 'defined':
        return df[features].to_numpy()
    elif feature_type == 'all':
        return np.hstack((fp_array, df[features].to_numpy()))
    else:
        raise ValueError(
            "Invalid feature_type. Choose 'fingerprints', 'defined', or 'all'."
        )


def objective(trial, X_train, y_train, X_val, y_val, model_name, cfg):
    """
    Objective function for Optuna optimization.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation target.
        model_name (str): Name of the model.
        cfg (DictConfig): Configuration object.

    Returns:
        float: RMSE score.
    """
    if model_name == 'random_forest':
        n_estimators = trial.suggest_int(
            'n_estimators', cfg.random_forest.n_estimators[0], cfg.random_forest.n_estimators[1])
        max_depth = trial.suggest_int(
            'max_depth', cfg.random_forest.max_depth[0], cfg.random_forest.max_depth[1])
        min_samples_split = trial.suggest_int(
            'min_samples_split', cfg.random_forest.min_samples_split[0], cfg.random_forest.min_samples_split[1])
        min_samples_leaf = trial.suggest_int(
            'min_samples_leaf', cfg.random_forest.min_samples_leaf[0], cfg.random_forest.min_samples_leaf[1])
        max_features = trial.suggest_categorical(
            'max_features', cfg.random_forest.max_features)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=cfg.random_state
        )
    elif model_name == 'lightgbm':
        num_leaves = trial.suggest_int(
            'num_leaves',
            cfg.lightgbm.num_leaves[0],
            cfg.lightgbm.num_leaves[1]
        )
        learning_rate = trial.suggest_float(
            'learning_rate',
            cfg.lightgbm.learning_rate[0],
            cfg.lightgbm.learning_rate[1],
            log=True
        )
        model = LGBMRegressor(num_leaves=num_leaves,
                              learning_rate=learning_rate)
    elif model_name == 'gaussian_process':
        alpha = trial.suggest_float(
            'alpha',
            cfg.gaussian_process.alpha[0],
            cfg.gaussian_process.alpha[1],
            log=True
        )
        model = GaussianProcessRegressor(alpha=alpha)
    elif model_name == 'xgboost':
        n_estimators = trial.suggest_int(
            'n_estimators',
            cfg.xgboost.n_estimators[0],
            cfg.xgboost.n_estimators[1]
        )
        max_depth = trial.suggest_int(
            'max_depth',
            cfg.xgboost.max_depth[0],
            cfg.xgboost.max_depth[1]
        )
        learning_rate = trial.suggest_float(
            'learning_rate',
            cfg.xgboost.learning_rate[0],
            cfg.xgboost.learning_rate[1],
            log=True
        )
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
        )
    else:
        raise ValueError(
            "Invalid model_name. Choose 'random_forest', 'lightgbm', "
            "'gaussian_process', or 'xgboost'."
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metric = calculate_metrics(
        y_val, y_pred, f"Optuna {model_name.replace('_', ' ').title()}"
    )
    return metric['mae']


def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, model_name, cfg, max_evals=50):
    """
    Train and evaluate a model using Optuna for hyperparameter optimization.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation target.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test target.
        model_name (str): Name of the model.
        cfg (DictConfig): Configuration object.
        max_evals (int): Maximum number of evaluations for Optuna.

    Returns:
        tuple: Trained model, train predictions, validation predictions, test predictions, metrics, and best hyperparameters.
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train,
                                X_val, y_val, model_name, cfg),
        n_trials=max_evals
    )

    best_trial = study.best_trial
    best_params = best_trial.params

    if model_name == 'random_forest':
        model = RandomForestRegressor(
            **best_params, random_state=cfg.random_state)
    elif model_name == 'lightgbm':
        model = LGBMRegressor(**best_params)
    elif model_name == 'gaussian_process':
        model = GaussianProcessRegressor(**best_params)
    elif model_name == 'xgboost':
        model = XGBRegressor(**best_params)

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    metrics = calculate_metrics(
        y_test, y_test_pred, f"Optuna {model_name.replace('_', ' ').title()}"
    )

    return model, y_train_pred, y_val_pred, y_test_pred, metrics, best_params


def save_results(output_dir, feature_type, model_name, metrics, train_predictions, val_predictions, test_predictions, best_params, seed):
    """
    Save model results to files.

    Args:
        output_dir (str): Directory to save results.
        feature_type (str): Type of features used.
        model_name (str): Name of the model.
        metrics (dict): Model performance metrics.
        train_predictions (pd.DataFrame): Training predictions.
        val_predictions (pd.DataFrame): Validation predictions.
        test_predictions (pd.DataFrame): Test predictions.
        best_params (dict): Best hyperparameters.
        seed (int): Random seed used.
    """
    with open(Path(output_dir, f'{model_name}_metrics_{feature_type}_seed_{seed}.json'), 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

    train_predictions.to_csv(Path(
        output_dir, f'{model_name}_train_predictions_{feature_type}_seed_{seed}.csv'), index=False)
    val_predictions.to_csv(Path(
        output_dir, f'{model_name}_val_predictions_{feature_type}_seed_{seed}.csv'), index=False)
    test_predictions.to_csv(Path(
        output_dir, f'{model_name}_test_predictions_{feature_type}_seed_{seed}.csv'), index=False)

    with open(Path(output_dir, f'{model_name}_best_params_{feature_type}_seed_{seed}.json'), 'w') as json_file:
        json.dump(best_params, json_file, indent=4)


def load_split_data(split_type, output_dir, cluster=None, cluster_number=None):
    """
    Load split data based on the split type.

    Args:
        split_type (str): Type of data split.
        output_dir (str): Directory containing split data.
        cluster (int): Cluster number for extrapolation for the csv file generated using kmeans_clustering.py.

    Returns:
        tuple: Train, validation and test data.
    """
    if split_type == 'interpolation':
        train_data = pd.read_csv(Path(output_dir, 'train_data.csv'))
        val_data = pd.read_csv(Path(output_dir, 'val_data.csv'))
        test_data = pd.read_csv(Path(output_dir, 'test_data.csv'))
    elif split_type == 'extrapolation':
        if cluster is None:
            raise ValueError(
                "For extrapolation, a cluster number must be provided.")
        train_data = pd.read_csv(
            Path(output_dir, f'train_data_extra_{cluster_number}_cluster_{cluster}.csv'))
        val_data = pd.read_csv(
            Path(output_dir, f'val_data_extra_{cluster_number}_cluster_{cluster}.csv'))
        test_data = pd.read_csv(
            Path(output_dir, f'test_data_extra_{cluster_number}_cluster_{cluster}.csv'))
    else:
        raise ValueError(
            "Invalid split_type. Choose 'interpolation' or 'extrapolation'.")

    return train_data, val_data, test_data


def get_extrapolation_clusters(output_dir, cluster_number):
    """
    Get the list of extrapolation clusters.

    Args:
        output_dir (str): Directory containing cluster files.

    Returns:
        list: List of cluster numbers.
    """
    cluster_files = Path(output_dir).glob(
        f'train_data_extra_{cluster_number}_cluster_*.csv')
    return sorted({int(f.stem.split('_')[-1]) for f in cluster_files})


def aggregate_metrics(metrics_list):
    """
    Aggregate metrics from multiple runs.

    Args:
        metrics_list (list): List of metric dictionaries.

    Returns:
        dict: Aggregated metrics.
    """
    aggregated_metrics = {}
    for key in metrics_list[0].keys():
        if key == 'model_name':
            aggregated_metrics[key] = metrics_list[0][key]
        else:
            try:
                values = [float(metrics[key]) for metrics in metrics_list]
                mean_value = statistics.mean(values)
                std_value = statistics.stdev(values) if len(values) > 1 else 0
                aggregated_metrics[key] = f"{mean_value:.4f} ± {std_value:.4f}"
            except (TypeError, ValueError) as e:
                print(f"Error processing metric '{key}': {e}")
                aggregated_metrics[key] = "N/A"
    return aggregated_metrics


def aggregate_final_metrics(output_dir, model_name, feature_type):
    """
    Aggregate final metrics from multiple aggregated JSON files for extrapolation.

    Args:
        output_dir (str): Directory containing aggregated JSON files.
        model_name (str): Name of the model.
        feature_type (str): Type of features used.

    Returns:
        dict: Final aggregated metrics.
    """
    aggregated_files = list(Path(output_dir).glob(
        f'{model_name}_metrics_{feature_type}_cluster_*_aggregated.json'))
    all_metrics = []
    for file in aggregated_files:
        with open(file, 'r') as json_file:
            metrics = json.load(json_file)
            all_metrics.append(metrics)

    final_aggregated_metrics = {}
    for key in all_metrics[0].keys():
        if key == 'model_name':
            final_aggregated_metrics[key] = all_metrics[0][key]
        else:
            try:
                values = []
                for metrics in all_metrics:
                    mean, std = metrics[key].split(' ± ')
                    values.append((float(mean), float(std)))

                mean_values = [v[0] for v in values]
                std_values = [v[1] for v in values]

                final_mean = statistics.mean(mean_values)
                final_std = statistics.mean(std_values)

                final_aggregated_metrics[key] = f"{final_mean:.4f} ± {final_std:.4f}"
            except (TypeError, ValueError) as e:
                print(f"Error processing metric '{key}': {e}")
                final_aggregated_metrics[key] = "N/A"

    with open(Path(output_dir, f'{model_name}_metrics_{feature_type}_final_aggregated_clusters.json'), 'w') as json_file:
        json.dump(final_aggregated_metrics, json_file, indent=4)

    return final_aggregated_metrics


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="config",
)
def main(cfg: DictConfig):
    """
    Main function to run the machine learning pipeline.

    Args:
        cfg (DictConfig): Configuration object.
    """
    data = load_data(cfg.data_path)
    features = cfg.features
    target = cfg.target

    if cfg.split_type == 'interpolation':
        output_dir = Path(cfg.output_dir, 'interpolation')
        train_data, val_data, test_data = interpolation_split_and_save_data(data, cfg)
    else:
        output_dir = Path(cfg.output_dir, 'extrapolation')
        if cfg.split_type == 'extrapolation':
            cluster_number = cfg.cluster_number
            clusters = get_extrapolation_clusters(output_dir, cluster_number)
            for cluster in clusters:
                print(f"Processing cluster {cluster}")
                train_data, val_data, test_data = load_split_data(cfg.split_type, output_dir, cluster, cluster_number)
                X_train = get_features(train_data, cfg, cfg.feature_type, features)
                X_val = get_features(val_data, cfg, cfg.feature_type, features)
                X_test = get_features(test_data, cfg, cfg.feature_type, features)
                y_train = train_data[target].to_numpy()
                y_val = val_data[target].to_numpy()
                y_test = test_data[target].to_numpy()

                print(f"Feature type: {cfg.feature_type}")
                print("Training set shape:", X_train.shape)
                print("Validation set shape:", X_val.shape)
                print("Test set shape:", X_test.shape)

                all_metrics = []
                for seed in cfg.seeds:
                    print(f"Running with seed {seed}")
                    np.random.seed(seed)
                    model, y_train_pred, y_val_pred, y_test_pred, metrics, best_params = train_and_evaluate_model(
                        X_train, y_train, X_val, y_val, X_test, y_test, model_name=cfg.model_name, cfg=cfg, max_evals=cfg.max_evals)

                    print(metrics)
                    print("Best model:", model)
                    print("Best hyperparameters:", best_params)

                    train_predictions = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
                    val_predictions = pd.DataFrame({'Actual': y_val, 'Predicted': y_val_pred})
                    test_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

                    save_results(output_dir, f"{cfg.feature_type}_cluster_{cluster}", cfg.model_name,
                                 metrics, train_predictions, val_predictions, test_predictions, best_params, seed)
                    all_metrics.append(metrics)

                aggregated_metrics = aggregate_metrics(all_metrics)
                with open(Path(output_dir, f'{cfg.model_name}_metrics_{cfg.feature_type}_cluster_{cluster}_aggregated.json'), 'w') as json_file:
                    json.dump(aggregated_metrics, json_file, indent=4)
                print(f"Aggregated metrics saved in {Path(output_dir, f'{cfg.model_name}_metrics_{cfg.feature_type}_cluster_{cluster}_aggregated.json')}")

            final_aggregated_metrics = aggregate_final_metrics(output_dir, cfg.model_name, cfg.feature_type)
            print(f"Final aggregated metrics: {final_aggregated_metrics}")
        else:
            X_train = get_features(train_data, cfg, cfg.feature_type, features)
            X_val = get_features(val_data, cfg, cfg.feature_type, features)
            X_test = get_features(test_data, cfg, cfg.feature_type, features)
            y_train = train_data[target].to_numpy()
            y_val = val_data[target].to_numpy()
            y_test = test_data[target].to_numpy()

            print(f"Feature type: {cfg.feature_type}")
            print("Training set shape:", X_train.shape)
            print("Validation set shape:", X_val.shape)
            print("Test set shape:", X_test.shape)

            all_metrics = []
            for seed in cfg.seeds:
                print(f"Running with seed {seed}")
                np.random.seed(seed)
                model, y_train_pred, y_val_pred, y_test_pred, metrics, best_params = train_and_evaluate_model(
                    X_train, y_train, X_val, y_val, X_test, y_test, model_name=cfg.model_name, cfg=cfg, max_evals=cfg.max_evals)

                print(metrics)
                print("Best model:", model)
                print("Best hyperparameters:", best_params)

                train_predictions = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
                val_predictions = pd.DataFrame({'Actual': y_val, 'Predicted': y_val_pred})
                test_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

                save_results(output_dir, cfg.feature_type, cfg.model_name, metrics,
                             train_predictions, val_predictions, test_predictions, best_params, seed)
                all_metrics.append(metrics)

            aggregated_metrics = aggregate_metrics(all_metrics)
            with open(Path(output_dir, f'{cfg.model_name}_metrics_{cfg.feature_type}_aggregated.json'), 'w') as json_file:
                json.dump(aggregated_metrics, json_file, indent=4)

            print(f"Aggregated metrics saved in {Path(output_dir, f'{cfg.model_name}_metrics_{cfg.feature_type}_aggregated.json')}")

if __name__ == "__main__":
    main()