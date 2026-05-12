import json
import logging
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    """Charge les données via DVC ou fallback sur sklearn en CI"""
    try:
        import dvc.api

        data_path = dvc.api.get_url(path="data/raw/diabetes.csv", repo=".")
        logger.info(f"Chargement DVC: {data_path}")
        return pd.read_csv(data_path)
    except Exception as e:
        logger.warning(
            f"DVC indisponible ({e}), utilisation du dataset sklearn"
        )
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df


def main():
    # Utilisation de SQLite pour éviter les warnings de dépréciation MLflow
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.sklearn.autolog()

    df = load_data()
    logger.info(f"Dataset shape: {df.shape}")

    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        logger.info(f"R2 Score: {score}")
        mlflow.log_metric("r2_score", score)

        # Récupération des infos MLflow
        run_id = run.info.run_id
        exp_id = run.info.experiment_id
        mlflow_link = f"http://mlflow-web:5000/#/experiments/{exp_id}/runs/{run_id}"

        # ✅ CRÉATION DU FICHIER METRICS.JSON (INDISPENSABLE POUR DISCORD)
        metrics = {
            "r2_score": round(float(score), 4),
            "model_type": "RandomForestRegressor",
            "run_id": run_id,
            "mlflow_link": mlflow_link,
            "author": "Nourhene Ben othmen 2ING DS-AI 2.2",
        }

        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("✅ Fichier metrics.json créé avec succès !")


if __name__ == "__main__":
    main()