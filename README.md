# HitPredict

**Project 21: Predicting Billboard Hits Using Spotify Data**  
UE24CS352A Machine Learning mini-project, E Section, October 2026.

| Team member | Student ID |
|---|---|
| N S RAJ NANJAPPA | PES1UG24CS287 |
| NAGARAJ HEGDE | PES1UG24CS288 |

This project independently implements a machine-learning pipeline using the original Stanford HitPredict dataset. It predicts the dataset's historical Billboard hit label from nine Spotify audio features. Source data, trained model, actual experiment outputs, an executed notebook, an offline Streamlit demo, a two-page PDF, and presentation slides are included.

## Start the demo

Use **Python 3.12**. Open a terminal inside this folder.

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Or activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Open the local address displayed by Streamlit, normally `http://localhost:8501`. Choose a held-out song example, select **Predict hit label**, and compare the prediction with the source label. You can also enter measured audio features or upload a CSV. No Spotify account, API token, or internet connection is needed after dependency installation. This demo does not accept a Spotify URL or an audio file directly.

The supplied model is trained already. Load only the trusted model shipped with this project. Retrain with the pinned environment if a dependency-version error occurs.

## Reproduce the experiment

```bash
python -m src.train
python -m pytest -q
python -m src.predict --input data/demo_tracks.csv --output predictions.csv
```

Training regenerates the model, metrics, predictions, figures, and processed data. Runtime depends on your computer. A single fixed seed (21) and single-worker estimators support repeatability. Small numerical differences across platforms are possible. If you deliberately alter the experiment, refresh the notebook and revise the submitted report and slides to match the new results.

Open `notebooks/HitPredict_Analysis.ipynb` in VS Code, Jupyter, or another notebook viewer. It includes executed outputs. The notebook reads the measured results and also supports optional retraining via its `RUN_TRAINING` flag. It never silently retrains on opening.

## Measured results

After removing 21 duplicate artist/title pairs, 4,019 tracks remain: 1,595 hits and 2,424 sampled non-hits. Artist names and identical audio vectors determine groups. Groups never cross the primary train/validation/test boundaries.

| Split | Tracks | Hits | Distinct groups |
|---|---:|---:|---:|
| Training | 2,554 | 1,011 | 1,395 |
| Validation | 641 | 268 | 349 |
| Test | 824 | 316 | 437 |

Five learned model families plus a majority baseline are compared over nine configurations. The random forest with 250 trees, maximum depth 16, and minimum leaf size 5 wins by mean 3-fold training group CV ROC-AUC (0.801). A validation F1 search chooses threshold 0.29. The saved model remains fitted only on training data, keeping its evaluated scores reproducible.

| Test metric | Selected threshold 0.29 | Reference threshold 0.50 | Majority baseline |
|---|---:|---:|---:|
| Accuracy | 70.87% | 74.76% | 61.65% |
| Precision | 57.69% | 68.24% | 0.00% |
| Recall | 90.19% | 63.92% | 0.00% |
| F1 | 0.704 | 0.660 | 0.000 |
| ROC-AUC | 0.814 | 0.814 | 0.500 |

At threshold 0.29 the confusion matrix is TN=299, FP=209, FN=31, TP=285. The lower cutoff prioritizes finding hits and produces many false positives. Threshold 0.50 is a fixed reference, not a test-optimized replacement. Average precision is 0.681 against test hit prevalence 0.383. The 500-resample group bootstrap gives ROC-AUC interval [0.773, 0.848]. This interval reflects held-out group resampling, not every source of uncertainty.

Training ROC-AUC is 0.982 versus test ROC-AUC 0.814, so overfitting remains. Instrumentalness, danceability, and acousticness have the largest validation permutation importance. Importance measures predictive association, not the causal effect of changing a song.

## Method and safeguards

1. Validate numeric ranges, labels, identities, missing values and duplicates. Preserve the raw CSVs with SHA-256 checksums.
2. Normalize artist/title case and whitespace. Remove duplicate song keys. Conflicting labels would be excluded and counted (none occurred).
3. Group by normalized artist and exact audio signatures. Reserve 20% of groups for testing, then 20% of the remaining groups for validation. Row proportions vary because artist discographies differ in size.
4. Fit median imputation and scaling inside each training fold. There are no missing audio values in this snapshot. Three zero-tempo entries remain as source values rather than invented measurements.
5. Compare logistic regression, shrinkage linear discriminant analysis, RBF SVM, random forest, and a one-hidden-layer neural network against a majority baseline. Select by training group CV ROC-AUC.
6. Choose the cutoff on validation F1 only. Evaluate the chosen pipeline once on the untouched primary test split. Report a separate row-split diagnostic without using it for model selection.

The row-split diagnostic shares 392 artists across its training/test sets and scores ROC-AUC 0.789. It uses a different test sample and more training rows, so the numerical difference is not a controlled estimate of leakage impact.

## Scope and limitations

- Labels come from the original authors, not an independently rebuilt Billboard database. Their report describes inconsistent collection/label windows. The main CSV has no release dates, so this experiment evaluates historical classification rather than prospective release-time forecasting.
- The included `ArtistScore` is excluded: reliable prior-hit timing cannot be audited from the no-date file. Key, mode, artist name and track title are also excluded to keep the nine-feature audio setup explicit.
- Different aliases or collaborations may still identify the same performer. Normalized names are a useful but imperfect grouping key.
- Sampled non-hits and the historical hit prevalence differ from the population of all new releases. A score of 0.80 is not an established 80% chance of becoming a future hit.
- Marketing, distribution, artist audience, playlist exposure and release timing are unobserved. No causal or investment conclusions follow from feature importance.
- The dated source CSV (3,221 rows) is preserved as a reference only and does not enter training. Dates would need validation before a credible time-based study.
- SVC's probability option emits a deprecation notice in the pinned scikit-learn version. Training completes successfully. Notices are recorded in `results/metrics.json`; the selected random forest is unaffected.

## Files

| Path | Purpose |
|---|---|
| `app.py` | Offline single-track and batch demo |
| `src/data.py` | Validation, cleaning, grouping and split creation |
| `src/train.py` | Model comparison, threshold choice, evaluation and saving |
| `src/predict.py` | Shared scoring logic and command-line batch tool |
| `src/plots.py` | Charts from measured results |
| `src/fetch_data.py` | Optional checksum-verified source recovery |
| `data/raw/` | Original CSVs and checksums |
| `data/processed/tracks.csv` | Cleaned records with split assignments |
| `models/hitpredict.joblib` | Trusted trained pipeline and metadata |
| `results/` | Metrics, CV results, predictions, errors and figures |
| `notebooks/HitPredict_Analysis.ipynb` | Executed explanatory analysis |
| `deliverables/HitPredict_Report.pdf` | Required two-page write-up |
| `deliverables/HitPredict_Presentation.pptx` | Editable presentation with speaker notes |
| `docs/DEMO_AND_VIVA.md` | Demo sequence and panel preparation |
| `docs/DATA_CARD.md` | Source attribution and label limitations |
| `docs/CONTRIBUTIONS.md` | Actual contribution record for the team to complete |
| `docs/SUBMISSION_CHECKLIST.md` | Remaining academic submission steps |

## Source credit

Elena Georgieva, Marcella Suta, and Nicholas Burton, [HitPredict, Stanford CS229 (2018)](https://cs229.stanford.edu/proj2018/report/16.pdf). Data is from the [public archive linked by that report](https://drive.google.com/file/d/1p13zKI5Q0QejLB-Q8jQBLRiJDRQNmGgD/view), retrieved 7 October 2026. We use its two CSVs and provide a new implementation. We do not include or execute the original authors' code, and their published performance is not represented as our measured result. Source attribution is retained; no new license is asserted over their dataset.

Method references: [group cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), [preventing leakage](https://scikit-learn.org/stable/common_pitfalls.html), and [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).

## Repository and submission

Keep this repository private and invite faculty/TAs through GitHub repository settings after their usernames are confirmed. Submit the PDF and slides through the faculty's required channel by **10 October 2026, 11:59 PM**. The assignment heading says “One-Page Write-up” but the explicit requirement says a two-page summary; this package follows the two-page requirement. Review dates are 5-9 October 2026.
