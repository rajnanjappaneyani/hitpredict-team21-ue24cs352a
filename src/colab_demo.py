"""Standalone notebook demo logic, also embedded as readable Colab notebook code."""
import base64
import hashlib
import io
import json
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

DEMO_FEATURES = ["Danceability", "Energy", "Loudness", "Speechiness", "Acousticness",
                 "Instrumentalness", "Liveness", "Valence", "Tempo"]

def decode_demo_assets(encoded, expected_hash):
    """Verify the embedded data snapshot before reading its source-defined splits."""
    packed = base64.b64decode(encoded)
    if hashlib.sha256(packed).hexdigest() != expected_hash:
        raise ValueError("The embedded data checksum does not match.")
    payload = json.loads(zlib.decompress(packed))
    tracks = pd.read_csv(io.StringIO(payload["tracks_csv"]))
    required = set(DEMO_FEATURES + ["row_id", "Artist", "Track", "Label", "group", "split"])
    if not required.issubset(tracks.columns):
        raise ValueError("The embedded dataset is incomplete.")
    if not tracks.row_id.is_unique or not tracks.Label.isin([0, 1]).all():
        raise ValueError("Invalid source identities or labels.")
    if set(tracks.split) != {"train", "validation", "test"}:
        raise ValueError("Unexpected split names.")
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if not set(tracks.loc[tracks.split == a, "group"]).isdisjoint(
                tracks.loc[tracks.split == b, "group"]):
            raise ValueError("An artist/audio group crosses evaluation splits.")
    return tracks, payload["reference"]

def fit_demo_model(tracks, reference):
    """Refit only the selected configuration on the original training partition."""
    train = tracks.loc[tracks.split == "train"]
    forest = RandomForestClassifier(n_estimators=250, max_depth=16,
        min_samples_leaf=5, random_state=21, n_jobs=1)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler()), ("model", forest)])
    model.fit(train[DEMO_FEATURES], train.Label)
    return {"model": model, "threshold": float(reference["threshold"]),
            "training_medians": train[DEMO_FEATURES].median().to_dict()}

def score_demo(frame, bundle):
    """Validate explicit inputs and score them with the current notebook model."""
    missing = set(DEMO_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    x = frame[DEMO_FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    if x.empty or not np.isfinite(x.to_numpy()).all():
        raise ValueError("Provide at least one track with finite numeric audio features.")
    for feature in DEMO_FEATURES:
        low, high = (-60., 5.) if feature == "Loudness" else ((0., 300.) if feature == "Tempo" else (0., 1.))
        if not x[feature].between(low, high).all():
            raise ValueError(f"{feature} must be between {low:g} and {high:g}.")
    result = frame.copy()
    result["hit_score"] = bundle["model"].predict_proba(x)[:, 1]
    result["predicted_label"] = (result.hit_score >= bundle["threshold"]).astype(int)
    return result

def evaluate_demo(tracks, bundle):
    test = tracks.loc[tracks.split == "test"]
    scored = score_demo(test, bundle)
    y, pred, score = scored.Label, scored.predicted_label, scored.hit_score
    metrics = {"accuracy": float(accuracy_score(y, pred)),
               "precision": float(precision_score(y, pred, zero_division=0)),
               "recall": float(recall_score(y, pred, zero_division=0)),
               "f1": float(f1_score(y, pred, zero_division=0)),
               "roc_auc": float(roc_auc_score(y, score)),
               "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}
    return scored, metrics

def create_demo_controls(examples, bundle):
    """Build notebook-native controls without a web server or external tunnel."""
    import ipywidgets as widgets
    from IPython.display import display

    examples = examples.reset_index(drop=True)
    options = [(f"{row.Artist} - {row.Track}", i) for i, row in examples.iterrows()]
    options.append(("Custom track", -1))
    selector = widgets.Dropdown(options=options, value=0, description="Track:",
                                layout=widgets.Layout(width="95%"))
    source_label = widgets.HTML()
    fields = {}
    for feature in DEMO_FEATURES:
        low, high, step = (-60., 5., .1) if feature == "Loudness" else ((0., 300., 1.) if feature == "Tempo" else (0., 1., .001))
        unit = " (dB)" if feature == "Loudness" else " (BPM)" if feature == "Tempo" else ""
        fields[feature] = widgets.BoundedFloatText(value=float(examples.iloc[0][feature]),
            min=low, max=high, step=step, description=feature + unit,
            style={"description_width": "160px"}, layout=widgets.Layout(width="330px"))
    output = widgets.Output()
    state = {"last_result": None, "last_error": None}
    def load_example(change=None):
        selected = selector.value
        values = bundle["training_medians"] if selected == -1 else examples.iloc[selected]
        for feature in DEMO_FEATURES:
            fields[feature].value = float(values[feature])
        source_label.value = ("Custom inputs start at training-set medians." if selected == -1 else
            "Source label for the original example: <b>" + ("Hit" if int(values.Label) else "Non-hit") +
            "</b>. Edits create a hypothetical track whose true outcome is unknown.")
        state.update(last_result=None, last_error=None)
        output.clear_output(wait=False)
    selector.observe(load_example, names="value")
    button = widgets.Button(description="Predict hit label", button_style="success",
                            layout=widgets.Layout(width="190px"))
    def predict(_=None):
        output.clear_output(wait=True)
        state.update(last_result=None, last_error=None)
        with output:
            try:
                row = pd.DataFrame([{name: widget.value for name, widget in fields.items()}])
                result = score_demo(row, bundle)
                state["last_result"] = result
                label = "Hit" if int(result.iloc[0].predicted_label) else "Non-hit"
                print(f"Predicted label: {label}")
                print(f"Model hit score: {result.iloc[0].hit_score:.3f}")
                print(f"Decision threshold: {bundle['threshold']:.2f}")
                print("This is a historical classification score, not a calibrated future-hit probability.")
            except (ValueError, TypeError) as exc:
                state["last_error"] = str(exc)
                print(f"Input error: {exc}")
    button.on_click(predict)
    load_example()
    controls = widgets.VBox([selector, source_label] + list(fields.values()) + [button, output])
    return {"view": controls, "selector": selector, "fields": fields,
            "button": button, "state": state, "predict": predict}

def create_batch_controls(examples, bundle):
    """Optional Colab upload/download buttons; displaying them never opens a picker."""
    import ipywidgets as widgets
    from IPython.display import display
    from pathlib import Path

    state = {"result": score_demo(examples, bundle)}
    output = widgets.Output()
    upload = widgets.Button(description="Upload track CSV", layout=widgets.Layout(width="180px"))
    download = widgets.Button(description="Download predictions", layout=widgets.Layout(width="200px"))
    def upload_csv(_):
        with output:
            output.clear_output(wait=True)
            try:
                from google.colab import files
                incoming = files.upload()
                if not incoming:
                    print("No file selected.")
                    return
                if len(incoming) != 1:
                    raise ValueError("Choose one CSV file at a time.")
                frame = pd.read_csv(io.BytesIO(next(iter(incoming.values()))))
                if len(frame) > 10000:
                    raise ValueError("Please upload at most 10,000 tracks.")
                result = score_demo(frame, bundle)
                state["result"] = result
                print(f"Scored {len(result)} tracks.")
                display(result)
            except ImportError:
                print("File picker available in Google Colab. In local Jupyter, use score_demo(pd.read_csv(path), bundle).")
            except (ValueError, TypeError) as exc:
                print(f"CSV error: {exc}")
    def download_csv(_):
        path = Path("hitpredict_predictions.csv")
        state["result"].to_csv(path, index=False)
        try:
            from google.colab import files
            files.download(str(path))
        except ImportError:
            with output:
                print(f"Saved {path}")
    upload.on_click(upload_csv)
    download.on_click(download_csv)
    return {"view": widgets.VBox([widgets.HBox([upload, download]), output]),
            "state": state, "upload": upload, "download": download}
