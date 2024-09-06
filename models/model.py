import os
import pickle
import datetime
from typing import List, Union
from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import optuna
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from polymetrix.polymer import Polymer
from polymetrix.featurizer import MultipleFeaturizer
from polymetrix.data_main import create_featurizer


class PolymerFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, featurizer: MultipleFeaturizer = None):
        self.featurizer = featurizer or create_featurizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.featurizer.featurize(polymer) for polymer in X])

    def featurize_many(self, polymers: List[Union[Polymer, str]]):
        processed_polymers = [
            Polymer.from_psmiles(p) if isinstance(p, str) else p for p in polymers
        ]
        return self.transform(processed_polymers)


class BaseModel(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.featurizer = PolymerFeaturizer(create_featurizer())
        self.regressor = self.create_regressor()
        self.pipeline = self._default_pipeline()

    @abstractmethod
    def create_regressor(self):
        pass

    def _default_pipeline(self):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", self.regressor),
            ]
        )

    def fit(self, polymers: List[Union[Polymer, str]], labels: List[float]):
        features = self.featurizer.featurize_many(polymers)
        self.pipeline.fit(features, labels)
        return self

    def predict(self, polymers: List[Union[Polymer, str]]):
        features = self.featurizer.featurize_many(polymers)
        return self.pipeline.predict(features)

    def get_featurizer_info(self):
        featurizer = self.featurizer.featurizer
        featurizer_details = []
        for f in featurizer.featurizers:
            details = {"name": f.__class__.__name__, "parameters": f.__dict__}
            featurizer_details.append(details)
        return {
            "total_featurizers": len(featurizer.featurizers),
            "featurizer_details": featurizer_details,
        }

    def save_pretrained(self, preset: str, model_dir: str = None):
        try:
            if model_dir is None:
                model_dir = self.get_default_model_dir()

            os.makedirs(model_dir, exist_ok=True)

            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{preset}_{current_time}.pkl"
            model_path = os.path.join(model_dir, model_filename)

            pretrained_data = {
                "featurizer": self.featurizer.featurizer,
                "pipeline": self.pipeline,
                "saved_at": current_time,
            }

            with open(model_path, "wb") as f:
                pickle.dump(pretrained_data, f)

            print(f"Model saved as {model_path}")
            return model_filename
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error saving model: {str(e)}")

    @staticmethod
    def get_default_model_dir():
        return os.path.join(os.getcwd(), "pretrained_models")


class RandomForestModel(BaseModel):
    def create_regressor(self):
        return RandomForestRegressor(**self.cfg.model)


class XGBoostModel(BaseModel):
    def create_regressor(self):
        return XGBRegressor(**self.cfg.model)


def load_data(cfg: DictConfig):
    df = pd.read_csv(cfg.data.input_file)
    polymers = df[cfg.data.polymer_column].apply(Polymer.from_psmiles).tolist()
    labels = df[cfg.data.label_column].tolist()
    return polymers, labels


def objective(trial, cfg: DictConfig, X_train, X_valid, y_train, y_valid):
    """
    Objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        cfg: Configuration dictionary
        X_train, X_valid: Training and validation feature sets
        y_train, y_valid: Training and validation target variables

    Returns:
        Evaluation metric for the trial
    """
    params = {}
    for param, value in cfg.model.items():
        if isinstance(value, DictConfig) and "min" in value and "max" in value:
            params[param] = (
                trial.suggest_int(param, value.min, value.max)
                if isinstance(value.min, int) and isinstance(value.max, int)
                else trial.suggest_float(param, value.min, value.max)
            )
        elif isinstance(value, list):
            params[param] = trial.suggest_categorical(param, value)

    model = create_model(cfg.model.name, params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    return mean_squared_error(y_valid, y_pred)


def create_model(model_name: str, params: dict):
    if model_name == "random_forest":
        return RandomForestModel(DictConfig({"model": params}))
    elif model_name == "xgboost":
        return XGBoostModel(DictConfig({"model": params}))
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def optimize_hyperparameters(cfg: DictConfig, X_train, X_valid, y_train, y_valid):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, cfg, X_train, X_valid, y_train, y_valid),
        n_trials=cfg.optimization.n_trials,
    )
    return study.best_params


def print_metrics(set_name: str, mse: float, mae: float, r2: float):
    print(f"{set_name} Set:")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared Score: {r2}")


def train_and_evaluate_model(
    cfg: DictConfig, X_train, X_valid, X_test, y_train, y_valid, y_test
):
    print(f"Optimizing hyperparameters for {cfg.model.name}...")
    best_params = optimize_hyperparameters(cfg, X_train, X_valid, y_train, y_valid)
    print(f"Best hyperparameters: {best_params}")

    cfg.model = OmegaConf.merge(cfg.model, best_params)
    model = create_model(cfg.model.name, cfg.model)

    featurizer_info = model.get_featurizer_info()
    print(f"Number of featurizers used: {featurizer_info['total_featurizers']}")
    print("Featurizers used:")
    for details in featurizer_info["featurizer_details"]:
        print(f"- {details['name']}")
        for param, value in details["parameters"].items():
            print(f"  {param}: {value}")
        print()

    model.fit(X_train, y_train)

    saved_filename = model.save_pretrained(f"my_polymer_model_{cfg.model.name}")

    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)

    valid_mse = mean_squared_error(y_valid, y_valid_pred)
    valid_mae = mean_absolute_error(y_valid, y_valid_pred)
    valid_r2 = r2_score(y_valid, y_valid_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Model Performance ({cfg.model.name}):")
    print_metrics("Validation", valid_mse, valid_mae, valid_r2)
    print()
    print_metrics("Test", test_mse, test_mae, test_r2)
    print()

    return best_params, model
