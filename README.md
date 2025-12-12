# Gold Price Forecasting Pipeline

This repository orchestrates a multi-stage pipeline to build a regime-aware gold price prediction model.

## Pipeline overview

Run `python run_project.py` to execute all stages sequentially. Each script can also be executed individually.

1. **01_data_pipeline.py** – Loads local gold price history from `data/XAU_USD Historical Data.csv`, fetches macroeconomic indicators from FRED, and merges them into `data/master_dataset.csv` with forward/backward filling for coverage.
2. **02_baseline_random_walk.py** – Evaluates a random-walk baseline on the master dataset and saves metrics/plots to `outputs/<RUN_ID>/`.
3. **02_2_statistical_diagnosis.py** – Runs diagnostic checks on the merged data (e.g., stationarity, correlations) and emits visualizations to the run output directory.
4. **03_feature_engineering.py** – Adds event-driven features (natural disasters, epidemics, GDELT signals) to create `data/model_ready_dataset.csv`.
5. **09_final_regime_boost.py** – Trains the gradient boosting model with regime interactions, evaluates performance, and logs metrics plus charts to `outputs/<RUN_ID>/`.

The `RUN_ID` environment variable controls the output subfolder (defaults to a timestamp) and is respected by all stages through `config.py`.

## Expected data layout

Place required CSV inputs in `data/` before running the pipeline:

- `XAU_USD Historical Data.csv` – Local gold price history with `Date` and `Price` columns.
- `natural_disasters.csv` – Catalog of disasters with start/end date fields; used for event flags.
- `epidemic_and_pandemics.csv` – Epidemic start dates mapped to the gold price timeline.
- `gdelt_daily_world_2013_present.csv` – GDELT-derived sentiment/conflict aggregates (must include a date column).

Generated artifacts:

- `data/master_dataset.csv` – Gold prices merged with FRED macro indicators.
- `data/model_ready_dataset.csv` – Feature-enriched dataset used by the final model.
- `outputs/<RUN_ID>/` – Metrics and plots for each stage (created automatically if missing).

## How to run

```bash
python run_project.py
```

Optional overrides via CLI flags or environment variables are described in `config.py` (e.g., `RUN_ID`, `DATA_DIR`, `TEST_SIZE_RATIO`).

## Troubleshooting

- **FRED data missing or fetch failures**: `01_data_pipeline.py` retries FRED downloads three times. Confirm internet connectivity and that the FRED service is reachable; the script will stop early if macro data cannot be fetched.
- **Network interruptions mid-run**: Re-run the pipeline after connectivity returns. Outputs are organized under `outputs/<RUN_ID>/`, so reruns with a new `RUN_ID` avoid overwriting previous results.
- **Missing local CSVs**: If `XAU_USD Historical Data.csv` or other inputs are absent, the corresponding stage will error (or skip optional sources). Verify filenames and placement in `data/`.

## Dependencies

Install pinned dependencies from `requirements.txt` (or the fully locked `requirements-lock.txt` if you prefer a frozen set):

```bash
pip install -r requirements.txt
```

To regenerate the lock file with `pip-tools`:

```bash
pip install pip-tools
pip-compile --output-file=requirements-lock.txt requirements.txt
```

### Runtime packages

- matplotlib==3.8.4
- numpy==1.26.4
- pandas==2.2.2
- pandas-datareader==0.10.0
- scikit-learn==1.4.2
- joblib==1.3.2
