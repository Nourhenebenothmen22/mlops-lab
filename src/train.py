import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes  # Fallback pour CI
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Charge les données via DVC ou génère des données de test en CI"""
    try:
        # Tentative DVC (fonctionne en local avec credentials valides)
        import dvc.api
        data_path = dvc.api.get_url(path='data/raw/diabetes.csv', repo='.')
        logger.info(f"Chargement DVC: {data_path}")
        return pd.read_csv(data_path)
    except Exception as e:
        logger.warning(f"DVC/S3 indisponible ({e}), utilisation du dataset de test sklearn")
        # Fallback pour CI : dataset intégré sklearn
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df

def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns"))
    mlflow.sklearn.autolog()
    
    df = load_data()
    logger.info(f"Dataset shape: {df.shape}")
    
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        logger.info(f"R2 Score: {score}")
        mlflow.log_metric("r2_score", score)

if __name__ == "__main__":
    main()