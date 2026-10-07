"""Batch scoring: python -m src.predict --input data/demo_tracks.csv --output predictions.csv"""
import argparse
import joblib
import pandas as pd
from .data import ROOT, validate_features

def load_bundle():
    # Load only the trusted project artifact, never a user-uploaded pickle.
    return joblib.load(ROOT/"models/hitpredict.joblib")

def score_frame(frame, bundle=None):
    bundle = bundle or load_bundle()
    x = validate_features(frame)
    p = bundle["model"].predict_proba(x)[:,1]
    result = frame.copy()
    result["hit_score"] = p
    result["predicted_label"] = (p >= bundle["threshold"]).astype(int)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",default="predictions.csv")
    args=parser.parse_args()
    try:
        result=score_frame(pd.read_csv(args.input))
    except (ValueError,FileNotFoundError) as e:
        parser.error(str(e))
    result.to_csv(args.output,index=False)
    print(f"Scored {len(result)} tracks. Saved {args.output}")

if __name__ == "__main__": main()
