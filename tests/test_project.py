import json
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from src.data import ROOT, FEATURES, load_clean, split_data, validate_features
from src.predict import load_bundle, score_frame

def test_no_song_artist_or_audio_overlap():
    d,audit=load_clean()
    assert audit["raw_rows"]==4040 and audit["duplicate_songs_removed"]==21
    assert d.song_key.is_unique
    splits=split_data(d)
    assert sorted(np.concatenate(list(splits.values())).tolist())==list(range(len(d)))
    for a,b in [("train","validation"),("train","test"),("validation","test")]:
        first,second=d.iloc[splits[a]],d.iloc[splits[b]]
        assert set(first.artist_key).isdisjoint(second.artist_key)
        assert set(map(tuple,first[FEATURES].to_numpy())).isdisjoint(map(tuple,second[FEATURES].to_numpy()))

def test_input_validation():
    good=pd.read_csv(ROOT/"data/demo_tracks.csv")
    for field,value in [("Danceability",1.1),("Tempo",np.inf),("Energy",np.nan),("Loudness",-100)]:
        bad=good.copy();bad.loc[0,field]=value
        with pytest.raises(ValueError):validate_features(bad)
    with pytest.raises(ValueError):validate_features(good.drop(columns="Tempo"))
    with pytest.raises(ValueError):validate_features(good.iloc[:0])

def test_saved_predictions_and_preprocessing():
    bundle=load_bundle()
    d,_=load_clean();s=split_data(d)
    test=d.iloc[s["test"]]
    result=score_frame(test,bundle)
    saved=pd.read_csv(ROOT/"results/test_predictions.csv")
    assert np.allclose(result.hit_score,saved.score)
    assert np.array_equal(result.predicted_label,saved.prediction)
    assert set(bundle["features"])==set(FEATURES)
    assert "ArtistScore" not in bundle["features"]
    train=d.iloc[s["train"]]
    assert np.allclose(bundle["model"].named_steps["imputer"].statistics_,train[FEATURES].median())
    assert np.allclose(bundle["model"].named_steps["scaler"].mean_,train[FEATURES].mean())

def test_metadata_does_not_affect_prediction():
    d=pd.read_csv(ROOT/"data/demo_tracks.csv")
    original=score_frame(d).hit_score
    d["Artist"]="An unseen artist";d["Track"]="An unseen title";d["Label"]=1-d.Label
    assert np.allclose(original,score_frame(d).hit_score)

def test_streamlit_prediction_flow():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(str(ROOT/"app.py"),default_timeout=30).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any(metric.label=="Model hit score" for metric in at.metric)
    at.selectbox[0].select(at.selectbox[0].options[1]).run()
    at.button[0].click().run()
    assert not at.exception
