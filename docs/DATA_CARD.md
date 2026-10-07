# Dataset card

## Provenance

Original work: Elena Georgieva, Marcella Suta, Nicholas Burton, Stanford CS229, 2018. Report: https://cs229.stanford.edu/proj2018/report/16.pdf

Archive linked in the report: https://tinyurl.com/yb6cvtek  
Resolved public archive: https://drive.google.com/file/d/1p13zKI5Q0QejLB-Q8jQBLRiJDRQNmGgD/view  
Retrieved: 7 October 2026. File-level hashes: `data/raw/checksums.json`.

Only `complete_project_data_no_date.csv` (4,040 rows) enters training. `complete_project_data.csv` (3,221 rows) is retained as a dated reference. They are overlapping views and must not be concatenated. No original source code or credentials from the archive are included.

The authors combined Billboard hits with sampled tracks from the Million Song Dataset and obtained Spotify audio features. Labels are inherited from that archive. The report's collection and label-window descriptions differ, so this implementation does not claim a verified exact label window. A 0 label means a sampled non-hit according to the source, not proof a song never charted anywhere or at any later date.

## Modeling fields

| Feature | Meaning | Unit / accepted range |
|---|---|---|
| Danceability | Suitability for dancing | 0-1 |
| Energy | Perceptual intensity | 0-1 |
| Loudness | Overall loudness | dB, demo accepts -60 to +5 |
| Speechiness | Presence of spoken content | 0-1 |
| Acousticness | Acoustic sound confidence | 0-1 |
| Instrumentalness | Likelihood of no vocal content | 0-1 |
| Liveness | Evidence of a live audience | 0-1 |
| Valence | Musical positivity | 0-1 |
| Tempo | Estimated pace | BPM, demo accepts 0-300 |

`Artist` and `Track` identify records; they are not predictors. `Label` is the target. `ArtistScore`, `Key` and `Mode` remain in source/processed data but are excluded from the model. `Year` and `Month` exist only in the dated reference CSV.

## Observed quality

4,019 cleaned rows, 2,181 normalized artists, 1,595 hit labels, 2,424 non-hit labels. Twenty-one duplicate artist/title pairs removed. No missing audio cells, invalid labels or conflicting duplicate labels found. Three tempo values are zero and retained, which may reflect failed tempo detection. Positive loudness values exist, so a blanket maximum of 0 dB would reject valid source records.

Grouping unites equal normalized artist strings and exact nine-feature vectors. Artist aliases, spelling differences, and collaborations are not fully resolved. The procedure reduces obvious overlap without asserting perfect performer identity resolution.

## Appropriate interpretation

Suitable for a historical classification mini-project and offline demonstration. The archive does not provide an auditable as-of-date observation protocol, ISRC-level identity resolution, the entire population of releases, or a prospective chart test. Forecasting requires new release-dated examples and labels defined over a fixed future observation window. No new license is asserted over the authors' source dataset.
