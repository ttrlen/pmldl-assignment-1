from argparse import ArgumentParser, Namespace
import json
import logging
import os
from pathlib import Path

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "true"

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


TARGET_COLUMN = "koi_disposition"
LOG_FEATURE_COLUMNS = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_insol",
    "koi_model_snr",
]
SCALED_FEATURE_COLUMNS = [
    "koi_teq",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
]
FEATURE_COLUMNS = [*LOG_FEATURE_COLUMNS, *SCALED_FEATURE_COLUMNS]


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("models/exoplanet_classifier.joblib"))
    parser.add_argument("--metrics-path", type=Path, default=Path("metrics.json"))
    parser.add_argument("--tracking-database-path", type=Path, default=Path("mlflow.db"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--regularization-strength", type=float, default=1.0)
    return parser.parse_args()


def build_pipeline(arguments: Namespace) -> Pipeline:
    feature_engineer = ColumnTransformer(
        transformers=[
            (
                "log_transform",
                Pipeline(
                    [
                        (
                            "log1p",
                            FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                LOG_FEATURE_COLUMNS,
            ),
            ("scale", StandardScaler(), SCALED_FEATURE_COLUMNS),
        ]
    )
    classifier = LogisticRegression(
        max_iter=arguments.max_iter,
        C=arguments.regularization_strength,
        class_weight="balanced",
        random_state=arguments.random_state,
    )
    return Pipeline([("feature_engineering", feature_engineer), ("classifier", classifier)])


def main() -> None:
    arguments = parse_arguments()
    train_dataframe = pd.read_csv(arguments.train_path)
    test_dataframe = pd.read_csv(arguments.test_path)
    model = build_pipeline(arguments)
    model.fit(train_dataframe[FEATURE_COLUMNS], train_dataframe[TARGET_COLUMN])
    predictions = model.predict(test_dataframe[FEATURE_COLUMNS])
    metrics = {
        "accuracy": accuracy_score(test_dataframe[TARGET_COLUMN], predictions),
        "f1_weighted": f1_score(test_dataframe[TARGET_COLUMN], predictions, average="weighted"),
        "f1_macro": f1_score(test_dataframe[TARGET_COLUMN], predictions, average="macro"),
    }
    report = classification_report(test_dataframe[TARGET_COLUMN], predictions, output_dict=True, zero_division=0)
    arguments.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, arguments.model_path)
    with arguments.metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    logging.getLogger("alembic").setLevel(logging.ERROR)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    mlflow.set_tracking_uri(f"sqlite:///{arguments.tracking_database_path.resolve()}")
    mlflow.set_experiment("exoplanet-classification")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "max_iter": arguments.max_iter,
                "regularization_strength": arguments.regularization_strength,
                "random_state": arguments.random_state,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict(report, "classification_report.json")
        mlflow.sklearn.log_model(model, name="exoplanet_classifier")
    print(f"Saved model and metrics: accuracy={metrics['accuracy']:.3f}, f1_weighted={metrics['f1_weighted']:.3f}")


if __name__ == "__main__":
    main()
