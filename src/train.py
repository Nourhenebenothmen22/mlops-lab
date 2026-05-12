import json
import logging
import os
import warnings
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    """Charge les données via DVC ou utilise le fallback sklearn en CI"""
    try:
        import dvc.api

        data_path = dvc.api.get_url(path="data/raw/diabetes.csv", repo=".")
        logger.info(f"Chargement DVC: {data_path}")
        return pd.read_csv(data_path)
    except Exception as e:
        logger.warning(
            f"DVC indisponible ({e}), utilisation du fallback sklearn"
        )
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    # =========================================================================
    # 📊 MODÈLE 1 : TABULAIRE (Random Forest -> DiabetesModel)
    # =========================================================================
    logger.info("--- 🚀 DÉMARRAGE MODÈLE 1 : DiabetesModel ---")
    exp_name_tab = "experiment_diabetes"
    mlflow.set_experiment(exp_name_tab)
    mlflow.sklearn.autolog()

    df = load_data()
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="rf_v1") as run_rf:
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=6, random_state=42
        )
        rf.fit(X_train, y_train)

        score = rf.score(X_test, y_test)
        logger.info(f"R2 Score: {score}")
        mlflow.log_metric("r2_score", score)

        run_id_rf = run_rf.info.run_id
        exp_id_rf = run_rf.info.experiment_id

        # Enregistrement dans le Registry
        try:
            mlflow.sklearn.log_model(
                rf, "model", registered_model_name="DiabetesModel"
            )
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            client.set_registered_model_alias(
                "DiabetesModel", "production", 1
            )
        except Exception as e:
            logger.warning(
                f"Info Registry (normal sur file store CI) : {e}"
            )

        # Extraction des prédictions d'échantillon
        preds = rf.predict(X_test[:3])
        logger.info(f"Prédictions de test : {preds}")

    # =========================================================================
    # 🧠 MODÈLE 2 : NLP (Transformers -> DistilBERT_Sentiment)
    # =========================================================================
    logger.info("--- 🚀 DÉMARRAGE MODÈLE 2 : DistilBERT_Sentiment ---")
    exp_name_nlp = "NLP_Transformers"
    mlflow.set_experiment(exp_name_nlp)

    nlp_status = "Échec"
    run_id_nlp = "N/A"
    try:
        import transformers

        logger.info(
            "⏳ Téléchargement du modèle Transformers (1-2 minutes)..."
        )
        pipeline = transformers.pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        logger.info("✅ Téléchargement terminé !")

        with mlflow.start_run(run_name="distilbert_sentiment_v1") as run_nlp:
            run_id_nlp = run_nlp.info.run_id
            try:
                mlflow.transformers.log_model(
                    transformers_model=pipeline,
                    artifact_path="model",
                    registered_model_name="DistilBERT_Sentiment",
                )
            except Exception as e:
                logger.warning(f"Info Registry NLP : {e}")

            logger.info(
                "🎉 SUCCÈS TOTAL ! Modèle NLP enregistré dans MLflow !"
            )
            nlp_status = "Enregistré avec succès"

    except Exception as e:
        logger.error(f"❌ Erreur NLP : {e}")
        nlp_status = f"Erreur ({e})"

    # =========================================================================
    # 💾 EXPORT COMPLET DES MÉTRIQUES POUR GITHUB ACTIONS & DISCORD
    # =========================================================================
    link_rf = f"http://mlflow-web:5000/#/experiments/{exp_id_rf}/runs/{run_id_rf}"

    metrics = {
        "tabular_model": "DiabetesModel",
        "tabular_run_name": "rf_v1",
        "r2_score": round(float(score), 4),
        "tabular_preds": [round(float(p), 2) for p in preds],
        "tabular_link": link_rf,
        "nlp_model": "DistilBERT_Sentiment",
        "nlp_run_name": "distilbert_sentiment_v1",
        "nlp_status": nlp_status,
        "author": "Nourhene Ben othmen 2ING DS-AI 2.2",
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"✅ Fichier metrics.json généré : {metrics}")


if __name__ == "__main__":
    main()