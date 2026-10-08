"""Build the self-contained Colab notebook from this project's committed data."""
from pathlib import Path
import base64
import hashlib
import json
import textwrap
import zlib
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
reference = json.loads((ROOT / "results/metrics.json").read_text())
assets = {"tracks_csv": (ROOT / "data/processed/tracks.csv").read_text(),
          "reference": {k: reference[k] for k in ["threshold", "test", "splits", "selected_model", "versions"]}}
packed = zlib.compress(json.dumps(assets, ensure_ascii=False).encode(), level=9)
encoded = base64.b64encode(packed).decode()
digest = hashlib.sha256(packed).hexdigest()
core = (ROOT / "src/colab_demo.py").read_text()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
notebook = nbf.v4.new_notebook()
notebook.cells = [
md("""# HitPredict: Google Colab demo
**Team 21 · UE24CS352A · E Section**  
N S RAJ NANJAPPA (PES1UG24CS287) and NAGARAJ HEGDE (PES1UG24CS288)

Run **Runtime > Run all**. The last cells display a song selector, editable audio features, a prediction button, and batch results. A standard CPU runtime is sufficient.

This notebook includes the cleaned dataset and original split assignments. It needs no GitHub token, Drive mount, or separate data upload. It fits the selected random forest inside the current runtime and keeps the original validation-selected cutoff of 0.29. All displayed runtime metrics are recomputed. Dependency versions can produce small differences from the submitted experiment.

The model classifies inherited historical Billboard labels. Its scores are not calibrated probabilities of future chart success."""),
md("## 1. Prepare the runtime"),
code("""import importlib.util
import subprocess
import sys

# Use the runtime's installed scientific libraries. Install only missing packages.
packages = {'numpy': 'numpy', 'pandas': 'pandas', 'sklearn': 'scikit-learn',
            'matplotlib': 'matplotlib', 'ipywidgets': 'ipywidgets'}
missing = [package for module, package in packages.items()
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', *missing])

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from IPython.display import display
try:
    from google.colab import output
    output.enable_custom_widget_manager()
except ImportError:
    pass
print('Python:', sys.version.split()[0])
print('scikit-learn:', sklearn.__version__)
print('Runtime ready. No GPU is required.')"""),
md("""## 2. Load the demo functions and embedded data
The functions below fit only the selected random forest configuration, validate input features, and provide notebook controls. The next folded cell contains a compressed data snapshot with a checksum. The dataset is the same 4,019 cleaned records used in the project, including the saved train/validation/test assignments."""),
code(core),
code("# Embedded source-data snapshot; rebuilding requires scripts/build_colab_demo.py.\nDEMO_ASSETS_B64 = (\n" +
     "\n".join('    "' + line + '"' for line in textwrap.wrap(encoded, 100)) +
     "\n)\nDEMO_ASSETS_SHA256 = " + repr(digest) + "\n", metadata={"cellView": "form", "colab": {"input_hidden": True}}),
code("""tracks, reference = decode_demo_assets(DEMO_ASSETS_B64, DEMO_ASSETS_SHA256)
summary = tracks.groupby('split').agg(tracks=('Label', 'size'), hits=('Label', 'sum'),
                                     groups=('group', 'nunique')).reindex(['train', 'validation', 'test'])
display(summary)
print('Loaded', len(tracks), 'tracks with verified group separation.')"""),
md("""## 3. Fit the selected model
This fits a random forest with 250 trees, maximum depth 16, and minimum leaf size 5 on the **2,554 training tracks only**. Imputation and scaling also fit only on training data. The original model comparison is documented in the analysis notebook. This demo does not repeat model selection or use test labels to tune its settings."""),
code("""from time import perf_counter
started = perf_counter()
bundle = fit_demo_model(tracks, reference)
print(f'Model ready in {perf_counter() - started:.1f} seconds.')
print(f'Decision threshold: {bundle["threshold"]:.2f}')"""),
md("""## 4. Evaluate on the held-out test tracks
“This runtime” reports newly computed metrics. “Original experiment” is the stored reference, not a claim that a different runtime must reproduce every value exactly. The confusion matrix rows are source labels and the columns are predictions."""),
code("""test_predictions, current_metrics = evaluate_demo(tracks, bundle)
metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
comparison = pd.DataFrame({
    'This runtime': {k: current_metrics[k] for k in metric_names},
    'Original experiment': {k: reference['test'][k] for k in metric_names}
})
display(comparison.round(4))
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(np.array(current_metrics['confusion_matrix']),
                       display_labels=['Non-hit', 'Hit']).plot(ax=axes[0], cmap='Greens', colorbar=False)
axes[0].set_title('Current runtime: test confusion matrix')
RocCurveDisplay.from_predictions(test_predictions.Label, test_predictions.hit_score, ax=axes[1])
axes[1].set_title('Current runtime: test ROC curve')
plt.tight_layout()
plt.show()"""),
md("""## 5. Interactive track demo
Choose a held-out track, or choose **Custom track** and enter measured audio features. Click **Predict hit label**. Editing an example creates a hypothetical input; its original label does not establish the outcome of the edited song."""),
code("""examples = tracks.loc[tracks.split == 'test'].groupby('Label', group_keys=False).head(4).copy()
demo_controls = create_demo_controls(examples, bundle)
display(demo_controls['view'])"""),
md("""## 6. Batch demo and CSV download
The sample batch contains eight held-out tracks, including mistakes if the model makes them. **Download predictions** saves the current batch. **Upload track CSV** optionally scores another file with the nine feature columns listed below; names and labels are never model inputs."""),
code("""print('Required CSV columns:', ','.join(DEMO_FEATURES))
sample_batch = score_demo(examples, bundle)
display(sample_batch[['Artist', 'Track', 'Label', 'hit_score', 'predicted_label']].round(3))
batch_controls = create_batch_controls(examples, bundle)
display(batch_controls['view'])"""),
md("""## 7. Editable-code fallback
If the widget controls do not appear, this cell produces a prediction directly. Change the values and run this cell again."""),
code("""custom_track = pd.DataFrame([{
    'Danceability': 0.65, 'Energy': 0.70, 'Loudness': -6.0,
    'Speechiness': 0.05, 'Acousticness': 0.15, 'Instrumentalness': 0.001,
    'Liveness': 0.12, 'Valence': 0.60, 'Tempo': 120.0
}])
display(score_demo(custom_track, bundle).round(4))"""),
md("""## Sources and scope
Dataset: Georgieva, Suta and Burton, [HitPredict, Stanford CS229 (2018)](https://cs229.stanford.edu/proj2018/report/16.pdf), using the public CSV archive linked in the report. The cleaned snapshot removes 21 duplicate artist/title pairs. The source date-window descriptions are inconsistent, and labels are inherited rather than independently reconstructed.

The model uses nine audio features. Artist names define evaluation groups, and ArtistScore is excluded. Aliases, historical sampling, omitted marketing factors, and uncalibrated scores limit generalization to new releases. Instrumentalness importance in the original analysis is an association, not a causal conclusion.

[Colab notebook upload and runtime documentation](https://research.google.com/colaboratory/faq.html). The notebook runs the interaction directly within Colab without a separate web server.""")]
notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"},
                     "colab": {"name": "HitPredict_Colab_Demo.ipynb", "provenance": []}}
out = ROOT / "notebooks/HitPredict_Colab_Demo.ipynb"
nbf.write(notebook, out)
print(f"Created {out.name}: {out.stat().st_size // 1024} KB, {sum(c.cell_type == 'code' for c in notebook.cells)} code cells.")
