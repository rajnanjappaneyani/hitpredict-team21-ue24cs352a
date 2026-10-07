# Demonstration and viva guide

## Before the review

Install dependencies and open the demo once before the session. Keep the PDF, slides and this guide available offline. Start the demo with `python -m streamlit run app.py --server.address 127.0.0.1`. Do not spend the live presentation installing packages or waiting for training.

## Suggested 7-minute presentation

1. **0:00-0:45, problem.** Explain binary historical hit-label classification and the nine audio inputs. Say the study uses a historical sample, so future chart success is a broader claim.
2. **0:45-1:45, data.** State 4,040 source rows, 21 duplicates removed, 4,019 usable rows. Explain what a source label means and why artist names are reserved for grouping.
3. **1:45-3:00, experiment.** Show train/validation/test counts and explain separate artist groups. Compare the five model families and majority baseline. Explain why test data did not choose the model or cutoff.
4. **3:00-4:00, results.** State ROC-AUC 0.814, F1 0.704 and recall 90.2% at threshold 0.29. Point to the 209 false positives. Discuss the recall/precision trade-off.
5. **4:00-5:30, live demo.** Select a held-out example, score it, and compare with its source label. Change one measured feature for a hypothetical scenario. Explain that this does not establish a causal change in chart success. Show the batch CSV example and results tab.
6. **5:30-7:00, conclusion.** Explain instrumentalness importance, overfitting, label/date limitations and the need for release-time data. Be ready to inspect code and answer questions.

## Live demo fallback

If the browser is unavailable, use:

```bash
python -m src.predict --input data/demo_tracks.csv --output predictions.csv
```

Open `predictions.csv` and explain `hit_score`, `predicted_label`, and the 0.29 threshold. The executed notebook and saved charts provide offline evidence of the completed run.

## Panel questions with answers

**What is the input and output?** Nine numeric Spotify audio features are the inputs. The model produces a score and a binary prediction for the source dataset's Billboard hit label.

**Is Spotify popularity the target?** No. The target is the original authors' Billboard label. Spotify popularity is not an input or label here.

**Why use classification instead of regression?** The outcome has two classes. We do not have a reliable continuous chart-rank or sales target.

**Why group by artist?** Songs by one artist can have related style or production. Keeping them in one split avoids evaluating primarily on familiar artists. Exact repeated audio vectors also stay within one group.

**Does grouping prove there is zero leakage?** No. It blocks known identities and exact audio duplicates across splits. Aliases and collaborations can still connect names; timing and source-label quality also remain limitations.

**What did you do with missing values?** None were found in this snapshot. The pipeline includes median imputation fitted only on each training fold for future missing source measurements. Demo uploads reject missing or invalid inputs instead of silently filling them.

**Why scale the inputs?** It helps logistic regression, SVM, LDA and the neural network compare features such as tempo and acousticness. Tree models do not need scaling, but the shared pipeline is harmless for them.

**How does logistic regression work?** It learns a weighted feature sum and maps it through the sigmoid function, p = 1 / (1 + exp(-z)). L2 regularization limits the size of weights. C is the inverse regularization strength.

**How does a random forest work?** It fits many trees on bootstrap samples with random feature subsets and averages their class scores. Limiting depth and setting a minimum leaf size help control memorization.

**Why did random forest win?** It had the highest mean training group CV ROC-AUC, 0.801, among the predeclared configurations. Its nonlinear interactions fit these audio data better by that selection criterion. This does not establish universal superiority.

**Why choose ROC-AUC?** It measures the model's ranking across thresholds. A random hit should tend to score above a random non-hit. Average precision, precision, recall and F1 complement it because the hit class is less common.

**Why is accuracy not enough?** Predicting every test track as a non-hit already gives 61.65% accuracy while finding zero hits. Recall, precision and F1 expose that failure.

**Why threshold 0.29 rather than 0.50?** An explicit validation search maximized F1 at 0.29. The lower threshold catches more hits but increases false positives. It was fixed before primary test evaluation. A marketing use case would need a separately justified error-cost objective.

**What are precision, recall and F1?** Precision = TP/(TP+FP), recall = TP/(TP+FN), and F1 = 2PR/(P+R). Here TP=285, FP=209 and FN=31 at the chosen threshold.

**Are the scores calibrated future-hit probabilities?** No. The sample has an artificial historical hit prevalence and limited coverage. We report model scores; real-world calibration would require representative data and external validation.

**How did you calculate confidence intervals?** We resampled whole held-out groups with replacement 500 times and recomputed metrics. The ROC-AUC interval is 0.773-0.848. It is conditional on this fitted model and split, not uncertainty over all possible datasets or training runs.

**What does permutation importance mean?** Shuffle one validation feature, keep the others unchanged, and measure the ROC-AUC decrease. A larger decline shows the fitted model relied on that feature. Correlated features can share importance. This is not causal evidence.

**Why exclude ArtistScore?** The no-date CSV cannot independently establish whether an artist's prior hit was known before each track's outcome. Removing the field avoids relying on an unauditable timing assumption.

**Does the model overfit?** Yes, to some extent: training ROC-AUC is 0.982 and test ROC-AUC is 0.814. Depth and leaf constraints reduce, but do not eliminate, overfitting. We report the gap rather than claim it is solved.

**Why does the random row split score lower?** Different samples, group composition and training sizes change the score. The diagnostic's 0.789 AUC and 392 overlapping artists do not establish a universal direction or magnitude of leakage bias.

**How does this differ from the 2018 paper?** It reuses the original CSV, but uses a new implementation with duplicate auditing, disjoint artist groups, an independent test set, fold-local preprocessing and explicit threshold tuning. Its results are not directly comparable to the paper's validation scores.

**What would improve the study?** Validate track identities and chart histories, define a hit within a fixed period after release, collect representative non-hits, use a temporal split, add only information available at prediction time, and assess calibration and subgroup stability.

**What is your individual contribution?** Answer from the work each member actually completed and understood. Update `docs/CONTRIBUTIONS.md` with real edits, checks and rehearsals. Do not attribute generated implementation work or another person's actions to yourself.

## Code walkthrough

Start at `src/data.py`: features, validation, duplicates, groups and split assignments. Then open `src/train.py`: candidates, training CV selection, validation cutoff, untouched test evaluation, model export. Finish at `src/predict.py` and `app.py`, which use the same scoring function. The tests check the important boundaries and exercise the app interaction.
