from pathlib import Path
import pandas as pd
from importlib import resources

def get_data_path(filename: str) -> Path:
    with resources.path('polymetrix.data', filename) as data_path:
        return data_path

def load_dataset(filename: str) -> pd.DataFrame:
    data_path = get_data_path(filename)
    return pd.read_csv(data_path)