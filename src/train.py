"""Run: python -m src.train. Model selection never consults test outcomes."""
import json
import hashlib
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss)
from sklearn.inspection import permutation_importance
from .data import ROOT, FEATURES, SEED, load_clean, split_data

def pipeline(model):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()), ("model", model)])

def candidates():
    return {
        "Majority baseline": pipeline(DummyClassifier(strategy="prior")),
        "Logistic regression C=0.1": pipeline(LogisticRegression(C=.1, max_iter=2000, random_state=SEED)),
        "Logistic regression C=1": pipeline(LogisticRegression(C=1, max_iter=2000, random_state=SEED)),
        "Linear discriminant analysis": pipeline(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "RBF SVM C=1": pipeline(SVC(C=1, kernel="rbf", probability=True, random_state=SEED)),
        "RBF SVM C=10": pipeline(SVC(C=10, kernel="rbf", probability=True, random_state=SEED)),
        "Random forest depth=8": pipeline(RandomForestClassifier(n_estimators=250, max_depth=8,
                     min_samples_leaf=5, random_state=SEED, n_jobs=1)),
        "Random forest depth=16": pipeline(RandomForestClassifier(n_estimators=250, max_depth=16,
                     min_samples_leaf=5, random_state=SEED, n_jobs=1)),
        "Neural network (16)": pipeline(MLPClassifier(hidden_layer_sizes=(16,), alpha=.1,
                     max_iter=700, random_state=SEED))}

def metrics(y, p, threshold):
    pred = np.asarray(p) >= threshold
    return {"accuracy": float(accuracy_score(y,pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y,pred)),
            "precision": float(precision_score(y,pred,zero_division=0)),
            "recall": float(recall_score(y,pred,zero_division=0)),
            "f1": float(f1_score(y,pred,zero_division=0)),
            "roc_auc": float(roc_auc_score(y,p)),
            "average_precision": float(average_precision_score(y,p)),
            "brier_score": float(brier_score_loss(y,p)),
            "confusion_matrix": confusion_matrix(y,pred,labels=[0,1]).tolist()}

def cluster_bootstrap(y, p, groups, threshold, repeats=500):
    rng = np.random.default_rng(SEED)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    mapping = {g: np.flatnonzero(groups == g) for g in unique}
    values = {"roc_auc": [], "f1": [], "accuracy": []}
    for _ in range(repeats):
        ix = np.concatenate([mapping[g] for g in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(y[ix])) < 2:
            continue
        m = metrics(y[ix], p[ix], threshold)
        for k in values:
            values[k].append(m[k])
    return {k: np.quantile(v,[.025,.975]).tolist() for k,v in values.items()}

def main():
    for folder in ["results/figures", "models", "data/processed"]:
        (ROOT/folder).mkdir(parents=True, exist_ok=True)
    d,audit = load_clean()
    ix = split_data(d)
    split_labels = np.full(len(d), "", dtype=object)
    for name, inds in ix.items(): split_labels[inds] = name
    d["split"] = split_labels
    d.to_csv(ROOT/"data/processed/tracks.csv", index=False)
    d[["row_id","Artist","Track","song_key","group","Label","split"]].to_csv(ROOT/"results/split_manifest.csv",index=False)
    train, val, test = (d.iloc[ix[n]] for n in ["train","validation","test"])
    cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(
        train[FEATURES], train.Label, train.group))
    rows, fitted, warning_log = [], {}, []
    for name, estimator in candidates().items():
        print(f"Training {name}", flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            auc = cross_val_score(estimator, train[FEATURES], train.Label,
                                  cv=cv, scoring="roc_auc", n_jobs=1, error_score="raise")
            estimator.fit(train[FEATURES], train.Label)
        warning_log.extend({"model":name,"message":str(w.message)} for w in caught)
        fitted[name] = estimator
        rows.append({"model":name,"cv_auc_mean":float(auc.mean()),"cv_auc_std":float(auc.std()),
                     **{f"fold_{i+1}_auc":float(v) for i,v in enumerate(auc)}})
    comparison = pd.DataFrame(rows).sort_values("cv_auc_mean",ascending=False,kind="stable")
    comparison.to_csv(ROOT/"results/model_comparison.csv",index=False)
    selected = comparison.iloc[0].model
    model = fitted[selected]
    val_p = model.predict_proba(val[FEATURES])[:,1]
    thresholds = np.arange(.10,.901,.01)
    choices = [(float(t),float(f1_score(val.Label,val_p>=t))) for t in thresholds]
    # Predefined tie-break: pick the best F1, then the cutoff nearest 0.5.
    threshold = sorted(choices,key=lambda z:(-z[1],abs(z[0]-.5),z[0]))[0][0]
    pd.DataFrame(choices,columns=["threshold","validation_f1"]).to_csv(ROOT/"results/threshold_search.csv",index=False)
    test_p = model.predict_proba(test[FEATURES])[:,1]
    test_metrics = metrics(test.Label,test_p,threshold)
    # Secondary diagnostic only: same chosen configuration under an easier row split.
    random_train, random_test = train_test_split(d,test_size=.2,stratify=d.Label,random_state=SEED)
    random_model = clone(model).fit(random_train[FEATURES],random_train.Label)
    random_p = random_model.predict_proba(random_test[FEATURES])[:,1]
    random_metrics = metrics(random_test.Label,random_p,.5)
    random_overlap = len(set(random_train.artist_key)&set(random_test.artist_key))
    baseline = fitted["Majority baseline"].predict_proba(test[FEATURES])[:,1]
    boot = cluster_bootstrap(test.Label.to_numpy(),test_p,test.group.to_numpy(),threshold)
    # Interpret on validation only. Test results do not drive feature selection.
    imp = permutation_importance(model,val[FEATURES],val.Label,scoring="roc_auc",
                                 n_repeats=15,random_state=SEED,n_jobs=1)
    importance = pd.DataFrame({"feature":FEATURES,"auc_drop_mean":imp.importances_mean,
                               "auc_drop_std":imp.importances_std}).sort_values("auc_drop_mean",ascending=False)
    importance.to_csv(ROOT/"results/feature_importance.csv",index=False)
    predictions = test[["row_id","Artist","Track","Label","group"]].copy()
    predictions["score"] = test_p
    predictions["prediction"] = (test_p >= threshold).astype(int)
    predictions["correct"] = predictions.Label == predictions.prediction
    predictions.to_csv(ROOT/"results/test_predictions.csv",index=False)
    errors = predictions[~predictions.correct].assign(confidence=lambda z:np.where(z.prediction==1,z.score,1-z.score))
    errors.sort_values("confidence",ascending=False).head(20).to_csv(ROOT/"results/error_examples.csv",index=False)
    # Hold-out examples are representative examples, not hand-picked successes.
    examples = test.groupby("Label",group_keys=False).head(4).copy()
    examples[["Artist","Track"]+FEATURES+["Label"]].to_csv(ROOT/"data/demo_tracks.csv",index=False)
    summary = {"created_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,
      "dataset":audit,"features":FEATURES,"selected_model":selected,"threshold":threshold,
      "selection":"Highest mean 3-fold artist/audio-group CV ROC-AUC on training set",
      "threshold_selection":"Maximum validation F1 over 0.10 through 0.90, step 0.01",
      "splits":{n:{"rows":len(inds),"hits":int(d.iloc[inds].Label.sum()),
                     "groups":int(d.iloc[inds].group.nunique())} for n,inds in ix.items()},
      "train":metrics(train.Label,model.predict_proba(train[FEATURES])[:,1],threshold),
      "validation":metrics(val.Label,val_p,threshold),"test":test_metrics,
      "test_at_0_5":metrics(test.Label,test_p,.5),"baseline_test":metrics(test.Label,baseline,.5),
      "test_cluster_bootstrap_95_ci":boot,
      "random_split_diagnostic":{"metrics_at_0_5":random_metrics,"overlapping_artists":random_overlap,
         "train_rows":len(random_train),"test_rows":len(random_test),
         "note":"Secondary diagnostic with more training data; not a controlled causal comparison or selection criterion."},
      "versions":{"python":platform.python_version(),"scikit_learn":sklearn.__version__,
                  "numpy":np.__version__,"pandas":pd.__version__},
      "raw_sha256":hashlib.sha256((ROOT/"data/raw/complete_project_data_no_date.csv").read_bytes()).hexdigest(),
      "warnings":warning_log}
    (ROOT/"results/metrics.json").write_text(json.dumps(summary,indent=2))
    joblib.dump({"model":model,"features":FEATURES,"threshold":threshold,
                 "model_name":selected,"training_medians":train[FEATURES].median().to_dict(),
                 "metrics":summary},ROOT/"models/hitpredict.joblib")
    from .plots import make_plots
    make_plots(d,comparison,predictions,importance,summary)
    print(json.dumps({"selected":selected,"test":test_metrics,"threshold":threshold},indent=2))

if __name__ == "__main__":
    main()
