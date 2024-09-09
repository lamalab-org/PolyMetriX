import os
import pickle
import datetime
from typing import List, Union, Dict, Any
from abc import ABC, abstractmethod
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
    def __init__(
        self,
        featurizer: MultipleFeaturizer = None,
        custom_featurizers: List[str] = None,
        use_default: bool = True,
    ):
        self.default_featurizer = (
            featurizer or create_featurizer() if use_default else None
        )
        self.custom_featurizers = custom_featurizers or []
        self.use_default = use_default

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []

        if self.use_default:
            default_features = np.array(
                [self.default_featurizer.featurize(polymer) for polymer in X]
            )
            features.append(default_features)

        if self.custom_featurizers:
            custom_features = np.array(
                [
                    [getattr(polymer, feat) for feat in self.custom_featurizers]
                    for polymer in X
                ]
            )
            features.append(custom_features)

        return np.hstack(features) if len(features) > 1 else features[0]

    def featurize_many(self, polymers: List[Union[Polymer, str]]):
        processed_polymers = [
            Polymer.from_psmiles(p) if isinstance(p, str) else p for p in polymers
        ]
        return self.transform(processed_polymers)


class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.featurizer = self.create_featurizer()
        self.regressor = self.create_regressor()
        self.pipeline = self._default_pipeline()

    def create_featurizer(self):
        featurizer_config = self.config.get("featurizer", {})
        custom_featurizers = featurizer_config.get("custom_featurizers", [])
        use_default = featurizer_config.get("use_default", True)
        if featurizer_config.get("type") == "default":
            return PolymerFeaturizer(
                create_featurizer() if use_default else None,
                custom_featurizers,
                use_default,
            )
        else:
            raise NotImplementedError(
                f"Featurizer type {featurizer_config.get('type')} not implemented"
            )

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
        info = {
            "use_default_featurizers": self.featurizer.use_default,
            "custom_featurizers": self.featurizer.custom_featurizers,
        }
        if self.featurizer.use_default:
            default_featurizer = self.featurizer.default_featurizer
            default_featurizer_details = []
            for f in default_featurizer.featurizers:
                details = {"name": f.__class__.__name__, "parameters": f.__dict__}
                default_featurizer_details.append(details)
            info["default_featurizer_details"] = default_featurizer_details
            info["total_default_featurizers"] = len(default_featurizer.featurizers)
        return info

    def save_pretrained(self, preset: str, model_dir: str = None):
        try:
            if model_dir is None:
                model_dir = self.get_default_model_dir()

            os.makedirs(model_dir, exist_ok=True)

            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{preset}_{current_time}.pkl"
            model_path = os.path.join(model_dir, model_filename)

            pretrained_data = {
                "featurizer": self.featurizer,
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
        return RandomForestRegressor(**self.config["model"])


class XGBoostModel(BaseModel):
    def create_regressor(self):
        return XGBRegressor(**self.config["model"])


def load_data(config: Dict[str, Any]):
    df = pd.read_csv(config["data"]["input_file"])
    polymers = df[config["data"]["polymer_column"]].apply(Polymer.from_psmiles).tolist()
    labels = df[config["data"]["label_column"]].tolist()

    custom_featurizers = config["featurizer"].get("custom_featurizers", [])
    for feat in custom_featurizers:
        if feat not in df.columns:
            raise ValueError(f"Custom featurizer '{feat}' not found in the input data.")

    for polymer, row in zip(polymers, df.itertuples()):
        for feat in custom_featurizers:
            setattr(polymer, feat, getattr(row, feat))

    return polymers, labels


def objective(trial, config: Dict[str, Any], X_train, X_valid, y_train, y_valid):
    params = {}
    for param, value in config["model"].items():
        if param != "name":
            if isinstance(value, dict) and "min" in value and "max" in value:
                params[param] = (
                    trial.suggest_int(param, value["min"], value["max"])
                    if isinstance(value["min"], int) and isinstance(value["max"], int)
                    else trial.suggest_float(param, value["min"], value["max"])
                )
            elif isinstance(value, list):
                params[param] = trial.suggest_categorical(param, value)

    trial_config = config.copy()
    trial_config["model"] = params
    trial_config["model"]["name"] = config["model"]["name"]

    model = create_model(config["model"]["name"], trial_config)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    return mean_squared_error(y_valid, y_pred)


def create_model(model_name: str, config: Dict[str, Any]):
    new_config = config.copy()
    new_config["model"] = {k: v for k, v in config["model"].items() if k != "name"}

    if model_name == "random_forest":
        return RandomForestModel(new_config)
    elif model_name == "xgboost":
        return XGBoostModel(new_config)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def optimize_hyperparameters(
    config: Dict[str, Any], X_train, X_valid, y_train, y_valid
):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, config, X_train, X_valid, y_train, y_valid),
        n_trials=config["optimization"]["n_trials"],
    )
    return study.best_params


def print_metrics(set_name: str, mse: float, mae: float, r2: float):
    print(f"{set_name} Set:")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared Score: {r2}")


def train_and_evaluate_model(
    config: Dict[str, Any], X_train, X_valid, X_test, y_train, y_valid, y_test
):
    print(f"Optimizing hyperparameters for {config['model']['name']}...")
    best_params = optimize_hyperparameters(config, X_train, X_valid, y_train, y_valid)
    print(f"Best hyperparameters: {best_params}")

    config["model"].update(best_params)
    model = create_model(config["model"]["name"], config)

    featurizer_info = model.get_featurizer_info()
    print("Featurizer Information:")
    print(f"Using default featurizers: {featurizer_info['use_default_featurizers']}")
    if featurizer_info["use_default_featurizers"]:
        print(
            f"Number of default featurizers: {featurizer_info['total_default_featurizers']}"
        )
        print("Default featurizers used:")
        for details in featurizer_info["default_featurizer_details"]:
            print(f"- {details['name']}")
    print(
        f"Custom featurizers used: {', '.join(featurizer_info['custom_featurizers'])}"
    )

    model.fit(X_train, y_train)

    saved_filename = model.save_pretrained(
        f"my_polymer_model_{config['model']['name']}"
    )

    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)

    valid_mse = mean_squared_error(y_valid, y_valid_pred)
    valid_mae = mean_absolute_error(y_valid, y_valid_pred)
    valid_r2 = r2_score(y_valid, y_valid_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Model Performance ({config['model']['name']}):")
    print_metrics("Validation", valid_mse, valid_mae, valid_r2)
    print()
    print_metrics("Test", test_mse, test_mae, test_r2)
    print()

    return best_params, model
