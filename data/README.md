# Data

The dataset is confidential and **must not be committed or redistributed** (it is git-ignored).

1. Download the shared OneDrive folder and put the series CSVs in `data/raw/`
   (any file names; one or more series per file; schema as in the brief).
2. `python -m training.prepare_data --src data/raw`

`prepare_data` checks the schema, sorts, drops exact duplicate rows, **skips byte-identical duplicate files**
(e.g. `series_A_1.csv` is identical to `series_A.csv` in the folder this was built from), refuses two *different*
files claiming the same `series_id`, and writes `data/processed/all_series.csv.gz` plus
`artifacts/data_manifest.json` (file hashes, row counts, missingness) for lineage.
