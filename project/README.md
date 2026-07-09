# CallIQ — Intelligent Call Prioritization System

A full-stack ML project that scores and ranks customer calls so agents work
the highest-value / highest-risk conversations first. Includes a trained
classifier, a FastAPI backend, and a single-page "clay" 3D dashboard.

## Run it (one command)

```
python run.py
```

That's it. The first run will:

1. Install any missing Python packages
2. Generate a synthetic training dataset (`data/data.csv`) — see note below
3. Run the preprocessing pipeline
4. Train Logistic Regression, Random Forest, and XGBoost, and keep the winner
5. Launch the server and open `http://127.0.0.1:8010/` in your browser

Every step is skipped automatically on the next run if its output already
exists, so re-running `python run.py` afterwards just starts the server in
under two seconds.

Windows users can double-click **`run_server.bat`**. macOS/Linux users can
run **`./run_server.sh`**. Both just call `run.py`.

## Why a synthetic dataset?

The original project needed a raw file at `data/data.csv` (a bank-marketing
style dataset with columns like `age`, `job`, `balance`, `campaign`,
`poutcome`, etc.) that was never included with the code — so on a fresh
machine, preprocessing and training had nothing to run on and the whole
pipeline broke before it even reached the API.

`generate_sample_data.py` produces a statistically realistic stand-in with
the exact same schema, so the entire pipeline runs immediately with zero
manual steps. **If you get the real dataset later**, just drop it at
`data/data.csv` (semicolon-separated, same columns), delete the
`data/processed/` and `models/` folders, and run `python run.py` again — it
will rebuild everything from the real data automatically.

## What's in the project

```
app.py                    FastAPI backend — all API endpoints
appp.html                 Single-page dashboard (served at "/")
main.py                   Thin re-export of app.py (for `uvicorn main:app`)
run.py                    One-command setup + launch script
generate_sample_data.py   Synthetic dataset generator
data_preprocessing.py     Cleaning, feature engineering, encoding, SMOTE
model_train.py            Trains & compares 3 models, saves the best one
priority_ranking_system.py  Standalone batch scoring/ranking script
requirements.txt          Python dependencies
run_server.bat / .sh      One-click launchers (Windows / macOS-Linux)
```

## The ML pipeline

**`data_preprocessing.py`**
- Loads the raw CSV, treats `"unknown"` as missing, imputes categorical
  columns with the mode and numeric columns with the median
- Engineers 5 interaction features: `total_contacts`, `contacted_before`,
  `previous_success`, `contact_intensity`, `engagement_score`
- One-hot encodes categoricals, scales everything with `StandardScaler`,
  and balances the training set with SMOTE
- Saves `data/processed/{X,y}_{train,test}.csv`, `models/scaler.pkl`, and
  `models/feature_columns.json` (the exact post-encoding column order)

**`model_train.py`**
- Trains Logistic Regression, Random Forest, and XGBoost
- Compares them on Accuracy / Precision / Recall / F1 / ROC-AUC
- Saves the winner as `models/best_model.pkl` (a fixed filename, so the API
  never has to guess which algorithm happened to win a given run)
- Saves `models/metrics.json` (full leaderboard) and
  `models/feature_importance.json` (top 15 features) for the dashboard's
  Insights tab

### Bugs fixed from the original code
- **Double-scaling bug**: `app.py` was calling `scaler.transform()` on
  `X_test.csv`, which is *already scaled* by the preprocessing step. That
  silently produced meaningless probabilities. Fixed — the API now feeds
  the already-scaled test data straight into the model.
- **`use_label_encoder=True` on XGBoost**: this parameter was removed in
  `xgboost >= 2.0` and throws a hard error on any recent install. Removed.
- **Hardcoded `XGBoost.pkl`**: if XGBoost didn't win, the API would crash
  looking for a file that was never written. The pipeline now always saves
  a canonical `best_model.pkl`.
- Various pandas `SettingWithCopyWarning` / chained-assignment issues in
  preprocessing, and a bool/int dtype mismatch from one-hot encoding, that
  could produce inconsistent behavior across pandas versions.

## The API (`app.py`)

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the dashboard |
| `/health` | GET | Model/scaler/data status |
| `/score`, `/predict` | POST | Rule-based priority score for one call |
| `/resolve-call` | POST | Priority + scripted next steps + CRM log template |
| `/call-list` | GET | ML-ranked call list from the held-out test batch (`?limit=`, `?bucket=`) |
| `/model-info` | GET | Active model, training leaderboard, feature importance |
| `/upload-score` | POST | Score **your own CSV** of callers with the rule engine — any column naming |

The rule-based engine (`/score`, `/predict`, `/resolve-call`,
`/upload-score`) never depends on the trained model, so it works instantly
even before training finishes, and can score any caller data you throw at
it. The ML engine (`/call-list`, `/model-info`) demonstrates the trained
classifier over the held-out test batch.

## The dashboard (`appp.html`)

A single HTML file, clay/claymorphism 3D theme, zero build step, zero
required external dependencies (works fully offline — the Google Font link
is a progressive enhancement only).

- **Dashboard** — animated stat cards, priority distribution + probability
  histogram, top-5 preview
- **Ranked calls** — full sortable/searchable list with High/Medium/Low
  filter chips and CSV export
- **Resolve call** — the triage form, with a one-click "copy to clipboard"
  for the CRM log
- **Upload & score** — drag-and-drop CSV scoring for your own caller lists,
  with flexible column matching and CSV export
- **Insights** — active model, full leaderboard vs. the two runner-up
  models, and a feature-importance chart
- Light/dark clay theme toggle (persisted), toast notifications instead of
  browser `alert()`s, and a responsive layout for mobile

## Notes on the "priority" scores

Two independent scoring paths exist by design:
1. **Rule-based** (`_baseline_score` in `app.py`) — transparent, deterministic,
   driven by severity / wait time / vulnerable / VIP flags. Used for
   `/predict`, `/resolve-call`, and any CSV you upload.
2. **ML-based** (`/call-list`) — the trained classifier's predicted
   probability of conversion/success, turned into a 0–100 score and a
   High/Medium/Low bucket, run over the held-out test split.

They're intentionally decoupled: the rule-based path works on any caller
you describe right now; the ML path demonstrates a trained model on data
shaped like its training set.
