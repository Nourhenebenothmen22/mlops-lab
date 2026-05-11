import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import dvc.api

# Chargement data via DVC (reproductible)
data_path = dvc.api.get_url(path='data/raw/diabetes.csv', repo='.')
df = pd.read_csv(data_path)
# ... suite du training avec autolog ...

mlflow.sklearn.autolog()
with mlflow.start_run():
    # ... entraînement ...
    pass