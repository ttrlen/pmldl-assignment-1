from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "koi_disposition"
FEATURE_COLUMNS = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
]


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=Path("data/raw/cumulative.csv"))
    parser.add_argument("--train-output-path", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--test-output-path", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outlier-contamination", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    dataframe = pd.read_csv(arguments.input_path, usecols=[TARGET_COLUMN, *FEATURE_COLUMNS])
    dataframe = dataframe.dropna(subset=[TARGET_COLUMN]).drop_duplicates().reset_index(drop=True)
    train_dataframe, test_dataframe = train_test_split(
        dataframe,
        test_size=arguments.test_size,
        random_state=arguments.random_state,
        stratify=dataframe[TARGET_COLUMN],
    )
    median_values = train_dataframe[FEATURE_COLUMNS].median()
    train_dataframe[FEATURE_COLUMNS] = train_dataframe[FEATURE_COLUMNS].fillna(median_values)
    test_dataframe[FEATURE_COLUMNS] = test_dataframe[FEATURE_COLUMNS].fillna(median_values)
    detector = IsolationForest(
        contamination=arguments.outlier_contamination,
        random_state=arguments.random_state,
    )
    train_inliers = detector.fit_predict(train_dataframe[FEATURE_COLUMNS]) == 1
    test_inliers = detector.predict(test_dataframe[FEATURE_COLUMNS]) == 1
    train_dataframe = train_dataframe.loc[train_inliers].reset_index(drop=True)
    test_dataframe = test_dataframe.loc[test_inliers].reset_index(drop=True)
    arguments.train_output_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.test_output_path.parent.mkdir(parents=True, exist_ok=True)
    train_dataframe.to_csv(arguments.train_output_path, index=False)
    test_dataframe.to_csv(arguments.test_output_path, index=False)
    print(f"Saved cleaned datasets: train={len(train_dataframe)}, test={len(test_dataframe)}")


if __name__ == "__main__":
    main()
