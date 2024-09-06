import os
import sys
import json
from typing import List, Dict
import numpy as np

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.model import load_data, train_and_evaluate_model


def run_experiment(cfg: DictConfig, seed: int) -> Dict:
    cfg.data.random_state = seed
    print(f"Running experiment with seed {seed}")
    polymers, labels = load_data(cfg)

    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        polymers, labels, test_size=cfg.data.test_size, random_state=seed
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_val, y_train_val, test_size=cfg.data.validation_size, random_state=seed
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_valid)}")
    print(f"Test samples: {len(X_test)}")

    best_params, model = train_and_evaluate_model(
        cfg, X_train, X_valid, X_test, y_train, y_valid, y_test
    )

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "seed": seed,
        "hyperparameters": best_params,
        "metrics": {"mse": mse, "mae": mae, "r2": r2},
    }


def calculate_average_metrics(results: List[Dict]) -> Dict:
    metrics = ["mse", "mae", "r2"]
    avg_metrics = {}
    for metric in metrics:
        values = [r["metrics"][metric] for r in results]
        avg_metrics[f"avg_{metric}"] = np.mean(values)
        avg_metrics[f"std_{metric}"] = np.std(values)

    # Average hyperparameters
    all_hyper_keys = set()
    for result in results:
        all_hyper_keys.update(result["hyperparameters"].keys())

    avg_hyperparameters = {}
    for key in all_hyper_keys:
        values = [r["hyperparameters"].get(key, np.nan) for r in results]
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            avg_hyperparameters[f"avg_{key}"] = np.mean(valid_values)
            avg_hyperparameters[f"std_{key}"] = (
                np.std(valid_values) if len(valid_values) > 1 else 0
            )
        else:
            avg_hyperparameters[f"avg_{key}"] = np.nan
            avg_hyperparameters[f"std_{key}"] = np.nan

    return {
        "average_metrics": avg_metrics,
        "average_hyperparameters": avg_hyperparameters,
    }


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Use seeds from the configuration
    seeds = cfg.experiment.seeds
    results = []

    for seed in seeds:
        result = run_experiment(cfg, seed)
        results.append(result)
        print(f"Seed {seed} results:")
        print(json.dumps(result, indent=2))
        print("\n" + "=" * 50 + "\n")

    average_results = calculate_average_metrics(results)

    # Combine all results
    final_results = {"individual_runs": results, "average_results": average_results}

    # Save results to JSON file
    output_dir = os.path.join(project_root, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{cfg.model.name}_multi_seed_results.json")
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"Results saved to {output_file}")
    print("\nAverage Results:")
    print(json.dumps(average_results, indent=2))


if __name__ == "__main__":
    main()