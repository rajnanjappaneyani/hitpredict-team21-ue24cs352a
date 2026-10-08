"""Exercise the portable notebook's data, model and interactive prediction path."""
import nbformat
import numpy as np
import pandas as pd
import pytest

from src.colab_demo import (DEMO_FEATURES, decode_demo_assets, fit_demo_model,
                            evaluate_demo, score_demo, create_demo_controls,
                            create_batch_controls)
from src.data import ROOT, FEATURES


@pytest.fixture(scope="module")
def demo():
    notebook = nbformat.read(ROOT / "notebooks/HitPredict_Colab_Demo.ipynb", as_version=4)
    nbformat.validate(notebook)
    embedded = next(c.source for c in notebook.cells
                    if c.cell_type == "code" and "DEMO_ASSETS_B64 =" in c.source)
    namespace = {}
    exec(embedded, namespace)
    tracks, reference = decode_demo_assets(namespace["DEMO_ASSETS_B64"],
                                            namespace["DEMO_ASSETS_SHA256"])
    return tracks, reference, fit_demo_model(tracks, reference), namespace, notebook


def test_notebook_contains_current_source_and_data(demo):
    tracks, _, _, namespace, notebook = demo
    assert DEMO_FEATURES == FEATURES
    pd.testing.assert_frame_equal(tracks, pd.read_csv(ROOT / "data/processed/tracks.csv"))
    core = (ROOT / "src/colab_demo.py").read_text()
    assert any(c.cell_type == "code" and c.source == core for c in notebook.cells)
    with pytest.raises(ValueError, match="checksum"):
        decode_demo_assets(namespace["DEMO_ASSETS_B64"], "0" * 64)


def test_colab_refit_reproduces_saved_predictions(demo):
    tracks, reference, bundle, _, _ = demo
    scored, metrics = evaluate_demo(tracks, bundle)
    actual = scored.set_index("row_id").sort_index()
    saved = pd.read_csv(ROOT / "results/test_predictions.csv").set_index("row_id").sort_index()
    np.testing.assert_allclose(actual.hit_score, saved.score, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(actual.predicted_label, saved.prediction)
    assert metrics["confusion_matrix"] == reference["test"]["confusion_matrix"]
    train = tracks.loc[tracks.split == "train"]
    np.testing.assert_allclose(bundle["model"].named_steps["imputer"].statistics_,
                               train[DEMO_FEATURES].median())


def test_notebook_prediction_button_and_custom_inputs(demo):
    tracks, _, bundle, _, _ = demo
    examples = tracks.loc[tracks.split == "test"].head(3)
    controls = create_demo_controls(examples, bundle)
    controls["selector"].value = 1
    controls["button"].click()
    assert controls["state"]["last_error"] is None
    expected = score_demo(examples.iloc[[1]], bundle).hit_score.iloc[0]
    assert controls["state"]["last_result"].hit_score.iloc[0] == pytest.approx(expected)
    controls["selector"].value = -1
    assert controls["state"]["last_result"] is None
    controls["fields"]["Tempo"].value = 120
    controls["button"].click()
    assert controls["state"]["last_error"] is None
    assert controls["state"]["last_result"].Tempo.iloc[0] == 120
    invalid = examples.copy()
    invalid.loc[invalid.index[0], "Energy"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        score_demo(invalid, bundle)
    invalid.loc[invalid.index[0], "Energy"] = 2
    with pytest.raises(ValueError, match="Energy must"):
        score_demo(invalid, bundle)


def test_colab_batch_download_matches_predictions(demo, tmp_path, monkeypatch):
    tracks, _, bundle, _, _ = demo
    examples = tracks.loc[tracks.split == "test"].head(3)
    controls = create_batch_controls(examples, bundle)
    monkeypatch.chdir(tmp_path)
    controls["download"].click()
    downloaded = pd.read_csv(tmp_path / "hitpredict_predictions.csv")
    np.testing.assert_allclose(downloaded.hit_score, score_demo(examples, bundle).hit_score)
