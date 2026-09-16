from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data/raw/cumulative.csv")
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

def main() -> None:
    dataframe = pd.read_csv(RAW_DATA_PATH, usecols=[TARGET_COLUMN, *FEATURE_COLUMNS])

    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")
    print("\nClass counts:")
    print(dataframe[TARGET_COLUMN].value_counts().to_string())
    print("\nMissing values by column:")
    print(dataframe.isna().sum().sort_values(ascending=False).to_string())
    print("\nNumeric summary:")
    summary = dataframe[FEATURE_COLUMNS].describe().T
    print(summary[["min", "25%", "50%", "75%", "max"]].to_string())


if __name__ == "__main__":
    main()
