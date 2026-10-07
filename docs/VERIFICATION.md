# Verification record

Completed on 7 October 2026 using the pinned environment.

- Training completed for all nine declared model configurations. Metrics and captured SVC deprecation notices are stored in `results/metrics.json`.
- Five automated tests passed: split separation, input validation, saved-model prediction/preprocessing consistency, metadata independence, and Streamlit prediction interactions.
- All ten analysis-notebook code cells executed successfully.
- The command-line predictor scored the eight supplied held-out examples and wrote a CSV. Successful execution does not mean every label prediction is correct.
- The report contains exactly two pages and was visually inspected after rendering.
- The presentation contains 12 slides with speaker notes, four editable tables and two editable charts. Package, font, geometry and chart-data checks passed. Every slide was rendered and visually inspected; no native PowerPoint application check is claimed.
- The local Streamlit server started successfully. A browser visual check was blocked because the browser tool could not verify its administrator-enforced access policy. The automated Streamlit interaction tests passed; no claim of a completed browser visual inspection is made.

Reproduce the code checks with `python -m pytest -q`. Reproduce the experiment with `python -m src.train`.
