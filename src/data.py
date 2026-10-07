"""Validate the source data and construct groups before any model fitting."""
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["Danceability", "Energy", "Loudness", "Speechiness", "Acousticness",
            "Instrumentalness", "Liveness", "Valence", "Tempo"]
UNIT_FEATURES = [x for x in FEATURES if x not in {"Loudness", "Tempo"}]
SEED = 21

def normalize(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).casefold()).strip()

def validate_features(frame, allow_missing=False):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    x = frame[FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    if x.empty:
        raise ValueError("Provide at least one track.")
    if np.isinf(x.to_numpy()).any() or (not allow_missing and x.isna().any().any()):
        raise ValueError("Audio features must be finite numbers with no missing values.")
    for name in UNIT_FEATURES:
        if ((x[name] < 0) | (x[name] > 1)).any():
            raise ValueError(f"{name} must be between 0 and 1.")
    if ((x.Tempo < 0) | (x.Tempo > 300)).any():
        raise ValueError("Tempo must be between 0 and 300 BPM (0 means unavailable in this source).")
    if ((x.Loudness < -60) | (x.Loudness > 5)).any():
        raise ValueError("Loudness must be between -60 and 5 dB.")
    return x

def load_clean(path=None):
    raw = pd.read_csv(path or ROOT / "data/raw/complete_project_data_no_date.csv")
    required = set(FEATURES + ["Artist", "Track", "Label"])
    if not required.issubset(raw.columns):
        raise ValueError(f"Source is missing {sorted(required - set(raw.columns))}")
    if raw[["Artist", "Track", "Label"]].isna().any().any():
        raise ValueError("Source identities and labels cannot be missing.")
    if not raw.Label.isin([0, 1]).all():
        raise ValueError("Source labels must be 0 or 1.")
    validate_features(raw, allow_missing=True)
    d = raw.copy()
    d["row_id"] = np.arange(len(d))
    d["artist_key"] = d.Artist.map(normalize)
    d["song_key"] = d.artist_key + " / " + d.Track.map(normalize)
    conflicts = d.groupby("song_key").Label.nunique()
    conflict_mask = d.song_key.isin(conflicts[conflicts > 1].index)
    excluded_conflicts = int(conflict_mask.sum())
    d = d.loc[~conflict_mask].copy()
    before = len(d)
    d = d.drop_duplicates("song_key", keep="first").reset_index(drop=True)

    # Unite identical artist names AND exact audio vectors. This prevents an
    # identical recording under another spelling from crossing split boundaries.
    parent = list(range(len(d)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(a)] = find(b)
    seen_artist, seen_audio = {}, {}
    for i, row in d.iterrows():
        for key, seen in [(row.artist_key, seen_artist), (tuple(row[FEATURES].fillna(-999)), seen_audio)]:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    d["group"] = [find(i) for i in range(len(d))]
    audit = {"raw_rows": len(raw), "clean_rows": len(d),
             "duplicate_songs_removed": before - len(d),
             "conflicting_label_rows_removed": excluded_conflicts,
             "missing_audio_cells": int(d[FEATURES].isna().sum().sum()),
             "zero_tempo_rows": int(d.Tempo.eq(0).sum()),
             "artists": int(d.artist_key.nunique()), "groups": int(d.group.nunique()),
             "class_counts": {str(k): int(v) for k,v in d.Label.value_counts().sort_index().items()}}
    return d, audit

def split_data(d):
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    dev, test = next(outer.split(d, d.Label, d.group))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED + 1)
    train_rel, val_rel = next(inner.split(d.iloc[dev], d.Label.iloc[dev], d.group.iloc[dev]))
    splits = {"train": dev[train_rel], "validation": dev[val_rel], "test": test}
    for name, ix in splits.items():
        if d.iloc[ix].Label.nunique() != 2:
            raise ValueError(f"{name} must contain both classes.")
    for a,b in [("train","validation"),("train","test"),("validation","test")]:
        assert set(d.iloc[splits[a]].group).isdisjoint(d.iloc[splits[b]].group)
    return splits
