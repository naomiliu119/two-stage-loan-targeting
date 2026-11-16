"""Load the public data and recreate the train/validation/test split."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

SOURCE_URL = "https://raw.githubusercontent.com/gsbDBI/ExperimentData/master/Advertising/adcontentworth_qje.csv"
SOURCE_SHA256 = "6491a4e5b816f1efdf2b55b43924465582a16301181cdfac2580fc9d0440b3c9"
TREATMENT, APPLICATION, LOAN = "oneln_trt", "applied", "tookup"
FEATURES = {"baseline": ["risk", "offer4", "wave"],
            "history": ["risk", "offer4", "wave", "trcount"]}
INDEX_DIGESTS = {
    "training": "1169c0eb37433bfc3deffab59dcd92429bae13e0608dc7539547eb149fec94c3",
    "test": "80758ac9350b873a8197eb48f52177b6b9a77a2f2f6094a850d8e3f052de45a9",
}


def index_digest(frame):
    return sha256(pd.util.hash_pandas_object(frame.index, index=False).to_numpy().tobytes()).hexdigest()


def load_data(path="data/raw/loan_marketing_experiment.csv"):
    """Download the CSV once and check it against the known checksum."""
    path = Path(path)
    if path.exists():
        content = path.read_bytes()
    else:
        with urlopen(SOURCE_URL, timeout=60) as response:
            content = response.read()
    if sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError("The downloaded CSV does not match the expected checksum.")
    raw = pd.read_csv(BytesIO(content))
    if raw.shape != (58168, 37):
        raise ValueError(f"Unexpected released shape: {raw.shape}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return raw


def split_data(raw):
    """Recreate the original split, keeping identical rows in the same set.

    Rows are grouped by a hash of all their values. The hash only keeps
    duplicate rows together; it is not used as a feature.
    """
    sample = raw.loc[raw["wave"].isin([2, 3])].copy()
    if len(sample) != 53194 or sample.index.has_duplicates:
        raise ValueError("Unexpected main sample or row index.")
    needed = list(dict.fromkeys(FEATURES["history"] + [TREATMENT, APPLICATION, LOAN]))
    if sample[needed].isna().any().any():
        raise ValueError("Missing values in model inputs or outcomes.")
    for name in [TREATMENT, APPLICATION, LOAN]:
        if not sample[name].isin([0, 1]).all():
            raise ValueError(f"Nonbinary {name}")
    if not (sample[LOAN] <= sample[APPLICATION]).all():
        raise ValueError("Found a loan without an application.")
    groups = pd.util.hash_pandas_object(sample, index=False)
    dev_pos, test_pos = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20261102).split(sample, groups=groups))
    dev = sample.iloc[dev_pos]
    train_pos, val_pos = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=20261103).split(dev, groups=groups.iloc[dev_pos]))
    partitions = {"training": dev.iloc[train_pos].copy(), "validation": dev.iloc[val_pos].copy(), "test": sample.iloc[test_pos].copy()}
    if tuple(map(len, partitions.values())) != (31916, 10639, 10639):
        raise ValueError("Partition sizes differ from the original split.")
    for name, expected in INDEX_DIGESTS.items():
        if index_digest(partitions[name]) != expected:
            raise ValueError(f"{name} rows differ from the original split.")
    group_sets = [set(pd.util.hash_pandas_object(f, index=False)) for f in partitions.values()]
    if any(group_sets[a] & group_sets[b] for a, b in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError("Duplicate rows ended up in different sets.")
    return partitions


def training_folds(train):
    groups = pd.util.hash_pandas_object(train, index=False).to_numpy()
    strata = train[["risk", "wave", TREATMENT, APPLICATION]].astype(str).agg("|".join, axis=1)
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261106).split(train, strata, groups))
    for fit, check in folds:
        if np.intersect1d(groups[fit], groups[check]).size:
            raise ValueError("Duplicate rows cross a CV fold.")
    return folds
