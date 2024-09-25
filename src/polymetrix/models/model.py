import os
import pickle
import datetime
from typing import List, Union, Dict, Any
from abc import ABC
import pandas as pd
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BaseModel(ABC):
    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
        self.pipeline = self._default_pipeline()
        self.test_metrics = {}

    def _default_pipeline(self):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", RandomForestRegressor()),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        features = X[self.feature_columns]
        self.pipeline.fit(features, y)
        return self

    def predict(self, X: pd.DataFrame):
        features = X[self.feature_columns]
        return self.pipeline.predict(features)

    def save_pretrained(self, preset: str, model_dir: str = None):
        if model_dir is None:
            model_dir = self.get_default_model_dir()

        os.makedirs(model_dir, exist_ok=True)

        model_type = self.pipeline.named_steps["regressor"].__class__.__name__.lower()

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{preset}_{model_type}_{current_time}.pkl"
        model_path = os.path.join(model_dir, model_filename)

        pretrained_data = {
            "feature_columns": self.feature_columns,
            "pipeline": self.pipeline,
            "test_metrics": self.test_metrics,
            "saved_at": current_time,
        }

        with open(model_path, "wb") as f:
            pickle.dump(pretrained_data, f)

        print(f"Model saved as {model_path}")
        return model_filename

    @staticmethod
    def get_default_model_dir():
        return os.path.join(os.getcwd(), "pretrained_models")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseModel":
        feature_columns = config["featurizer"]["custom_featurizers"]
        model = cls(feature_columns)
        regressor_params = {k: v for k, v in config["model"].items() if k != "name"}
        model.pipeline.set_params(
            **{"regressor__" + k: v for k, v in regressor_params.items()}
        )
        return model

    @classmethod
    def from_pretrained(cls, model_path: Union[str, os.PathLike]) -> "BaseModel":
        with open(model_path, "rb") as handle:
            d = pickle.load(handle)

        model = cls(feature_columns=d["feature_columns"])
        model.pipeline = d["pipeline"]
        model.test_metrics = d.get("test_metrics", {})
        return model


def load_data(config: Dict[str, Any]):
    df = pd.read_csv(config["data"]["input_file"])
    X = df.drop(columns=[config["data"]["label_column"]])
    y = df[config["data"]["label_column"]]

    custom_featurizers = config["featurizer"]["custom_featurizers"]
    for feat in custom_featurizers:
        if feat not in df.columns:
            raise ValueError(f"Custom featurizer '{feat}' not found in the input data.")

    return X, y


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

    model_config = config.copy()
    model_config["model"].update(params)
    model = BaseModel.from_config(model_config)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    return mean_squared_error(y_valid, y_pred)


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
    print(f"Optimizing hyperparameters...")
    best_params = optimize_hyperparameters(config, X_train, X_valid, y_train, y_valid)
    print(f"Best hyperparameters: {best_params}")

    config["model"].update(best_params)
    model = BaseModel.from_config(config)

    print("Feature columns used:")
    print(", ".join(model.feature_columns))

    model.fit(X_train, y_train)

    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)

    valid_mse = mean_squared_error(y_valid, y_valid_pred)
    valid_mae = mean_absolute_error(y_valid, y_valid_pred)
    valid_r2 = r2_score(y_valid, y_valid_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Model Performance:")
    print_metrics("Validation", valid_mse, valid_mae, valid_r2)
    print()
    print_metrics("Test", test_mse, test_mae, test_r2)
    print()

    # Store test metrics in the model object
    model.test_metrics = {"mse": test_mse, "mae": test_mae, "r2": test_r2}

    return best_params, model
