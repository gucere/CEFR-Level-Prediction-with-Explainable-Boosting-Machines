"""Length-dependent features, read from the two section-1 length lists."""
import csv
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[2]
LENGTH_LISTS = (
    MODULE_DIR / "blacklist_1_direct_length_features.csv",
    MODULE_DIR / "blacklist_2_length_correlated_features.csv",
)
LENGTH_NOTE = (
    "Length-dependent features are the features on either section-1 length list: "
    "blacklist_1_direct_length_features.csv (numbers of words and different words) and "
    "blacklist_2_length_correlated_features.csv (other raw counts, and measures that "
    "still follow text length within course levels)."
)


def length_features(lists=LENGTH_LISTS):
    """Casefolded names of the listed features by source; every analysis excludes both lists."""
    features = {}
    for path in lists:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing length list: {path}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or not {"source", "feature"}.issubset(rows[0]):
            raise ValueError(f"{path.name} needs source and feature columns and at least one row.")
        for row in rows:
            features.setdefault(row["source"].strip().upper(), set()).add(row["feature"].strip().casefold())
    return features
