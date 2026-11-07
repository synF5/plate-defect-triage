"""Load the steel plates data and split it into train and test."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_VALUES = REPO_ROOT / "data" / "raw" / "Faults.NNA"
RAW_NAMES = REPO_ROOT / "data" / "raw" / "Faults27x7_var"
HOLDOUT_CSV = REPO_ROOT / "test_data.csv"

TARGET = "fault_type"
HOLDOUT_SHARE = 0.20
# last 4 digits of my BITS ID
SPLIT_SEED = 5781

# the label is given as 7 one-hot columns at the end of each row
FAULT_LABELS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]


def load_plates() -> pd.DataFrame:
    names = [line.strip() for line in RAW_NAMES.read_text().splitlines() if line.strip()]
    table = pd.read_csv(RAW_VALUES, sep=r"\s+", header=None, names=names)

    one_hot = table[FAULT_LABELS]
    if not (one_hot.sum(axis=1) == 1).all():
        raise ValueError("Every plate should carry exactly one fault label.")

    plates = table.drop(columns=FAULT_LABELS)
    plates[TARGET] = one_hot.idxmax(axis=1)
    return plates


def measurement_columns() -> list[str]:
    return [column for column in load_plates().columns if column != TARGET]


def split_plates(plates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # stratify so the small classes (Dirtiness has only 55) end up in test too
    return train_test_split(
        plates,
        test_size=HOLDOUT_SHARE,
        stratify=plates[TARGET],
        random_state=SPLIT_SEED,
    )


if __name__ == "__main__":
    train_part, holdout_part = split_plates(load_plates())
    holdout_part.to_csv(HOLDOUT_CSV, index=False)
    print(f"train={len(train_part)} holdout={len(holdout_part)} -> {HOLDOUT_CSV.name}")
    print(holdout_part[TARGET].value_counts().to_string())
