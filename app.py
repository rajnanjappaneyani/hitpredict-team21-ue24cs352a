"""Offline Streamlit demonstration. Start with: streamlit run app.py"""
import json
import pandas as pd
import streamlit as st
from src.data import ROOT, FEATURES
from src.predict import load_bundle, score_frame

st.set_page_config(page_title="HitPredict · Team 21", page_icon="🎵", layout="wide")
st.title("HitPredict")
st.write("Predicting Billboard hit labels from Spotify audio features")
st.caption("Team 21 · UE24CS352A · E Section · N S RAJ NANJAPPA and NAGARAJ HEGDE")

@st.cache_resource
def resources():
    return load_bundle()

if not (ROOT/"models/hitpredict.joblib").exists():
    st.error("The trained model is missing. Run python -m src.train from the project folder.")
    st.stop()
bundle = resources()
m = bundle["metrics"]
st.info("This model classifies historical dataset labels. Its score is not a calibrated chance of future Billboard success.")
predict_tab, batch_tab, results_tab, data_tab = st.tabs(["Predict a track", "Score a CSV", "Model results", "Dataset and method"])

with predict_tab:
    examples = pd.read_csv(ROOT/"data/demo_tracks.csv")
    labels = [f"{r.Artist} — {r.Track}" for r in examples.itertuples()]
    chosen = st.selectbox("Load a held-out example or enter your own features", ["Custom track"]+labels)
    defaults = bundle["training_medians"] if chosen=="Custom track" else examples.iloc[labels.index(chosen)].to_dict()
    st.caption("Custom tracks start at training-set medians. Enter measured audio features before interpreting a score.")
    with st.form("track_form"):
        columns = st.columns(3)
        values = {}
        for i,feature in enumerate(FEATURES):
            lo,hi,step = (-60.,5.,.1) if feature=="Loudness" else ((0.,300.,1.) if feature=="Tempo" else (0.,1.,.001))
            unit = " (dB)" if feature=="Loudness" else " (BPM)" if feature=="Tempo" else ""
            with columns[i%3]:
                values[feature] = st.number_input(feature+unit,min_value=lo,max_value=hi,
                    value=float(defaults[feature]),step=step,format="%.3f",key=f"{chosen}_{feature}")
        submitted=st.form_submit_button("Predict hit label", type="primary")
    if submitted:
        scored=score_frame(pd.DataFrame([values]),bundle).iloc[0]
        col1,col2=st.columns(2)
        col1.metric("Model hit score",f"{scored.hit_score:.3f}")
        col2.metric("Predicted label", "Hit" if scored.predicted_label else "Non-hit")
        st.write(f"Decision threshold: {bundle['threshold']:.2f}. Selected model: {bundle['model_name']}.")
        if chosen!="Custom track":
            actual=int(examples.iloc[labels.index(chosen)].Label)
            st.write("Source label for this held-out example: **"+("Hit" if actual else "Non-hit")+"**.")
        st.caption("Artist and track names are display metadata and are never model inputs.")

with batch_tab:
    st.write("Upload a CSV with these nine numeric columns:")
    st.code(",".join(FEATURES), language=None)
    st.download_button("Download example CSV",examples.to_csv(index=False),"demo_tracks.csv","text/csv")
    uploaded=st.file_uploader("Track feature CSV",type=["csv"])
    if uploaded is not None:
        try:
            incoming=pd.read_csv(uploaded)
            if len(incoming)>10000: raise ValueError("Please upload at most 10,000 rows at a time.")
            scored=score_frame(incoming,bundle)
            st.dataframe(scored,hide_index=True)
            st.download_button("Download predictions",scored.to_csv(index=False),"hitpredict_scores.csv","text/csv")
        except (ValueError,TypeError) as e:
            st.error(str(e))

with results_tab:
    st.subheader("Results on artists kept out of training")
    cols=st.columns(4)
    for col,key,label in zip(cols,["accuracy","precision","recall","roc_auc"],["Accuracy","Precision","Recall","ROC-AUC"]):
        col.metric(label,f"{m['test'][key]:.3f}")
    st.write(f"{m['splits']['test']['rows']} test tracks. F1: {m['test']['f1']:.3f}. Majority-baseline accuracy: {m['baseline_test']['accuracy']:.3f}.")
    st.caption("Model chosen by training group cross-validation. Threshold chosen on validation data. Test data was reserved for final evaluation.")
    st.dataframe(pd.read_csv(ROOT/"results/model_comparison.csv"),hide_index=True)
    st.image(str(ROOT/"results/figures/test_curves.png"))
    left,right=st.columns(2)
    left.image(str(ROOT/"results/figures/confusion_matrix.png"))
    right.image(str(ROOT/"results/figures/feature_importance.png"))

with data_tab:
    st.write(f"The original Stanford HitPredict archive contains {m['dataset']['raw_rows']:,} tracks in the no-date CSV. Cleaning retains {m['dataset']['clean_rows']:,} tracks.")
    st.write("Labels are inherited from the authors: 1 denotes a Billboard hit in their study and 0 denotes a sampled non-hit. We do not independently reconstruct the chart histories.")
    st.image(str(ROOT/"results/figures/data_splits.png"))
    st.write("Nine audio features feed an imputer, scaler, and classifier. Training, validation, and test use separate artist/audio groups. ArtistScore, names, and chart-related fields are excluded from prediction.")
    st.write("Limitations: the study has inconsistent date-window descriptions, the no-date file cannot support a release-time forecast, and aliases or collaborations may still connect different artist names. Sampled non-hits may not represent today's release population.")
    st.link_button("Original Stanford report", "https://cs229.stanford.edu/proj2018/report/16.pdf")
