"""
Generates a realistic, synthetic "call campaign" dataset shaped like the
classic bank-marketing dataset that this project's preprocessing pipeline
(data_preprocessing.py) expects.

Why this file exists
---------------------
The original project needs a file at data/data.csv (semicolon separated)
to train a model. That raw file was never shared. Without it, nothing
downstream (preprocessing -> training -> serving) can run, and the whole
project breaks on a fresh machine.

This script produces a synthetic but statistically believable dataset with
the same columns, so `run.py` can bootstrap the entire pipeline on any
machine with one command, with zero manual steps and zero external
downloads.

If you later get the real dataset, just drop it at data/data.csv
(semicolon-separated, same columns) and delete data/processed + models/ —
run.py will rebuild everything from the real data automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

JOBS = ["admin.", "technician", "services", "management", "retired",
        "blue-collar", "unemployed", "entrepreneur", "housemaid",
        "self-employed", "student"]
MARITAL = ["married", "single", "divorced"]
EDUCATION = ["primary", "secondary", "tertiary", "unknown"]
CONTACT = ["cellular", "telephone", "unknown"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
          "sep", "oct", "nov", "dec"]
POUTCOME = ["success", "failure", "other", "unknown"]


def generate(n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 90, n_rows)
    job = rng.choice(JOBS, n_rows)
    marital = rng.choice(MARITAL, n_rows, p=[0.55, 0.30, 0.15])
    education = rng.choice(EDUCATION, n_rows, p=[0.18, 0.5, 0.27, 0.05])
    default = rng.choice(["yes", "no"], n_rows, p=[0.02, 0.98])
    balance = rng.normal(1400, 3000, n_rows).astype(int)
    housing = rng.choice(["yes", "no"], n_rows, p=[0.55, 0.45])
    loan = rng.choice(["yes", "no"], n_rows, p=[0.16, 0.84])
    contact = rng.choice(CONTACT, n_rows, p=[0.65, 0.10, 0.25])
    day = rng.integers(1, 29, n_rows)
    month = rng.choice(MONTHS, n_rows)
    duration = np.abs(rng.normal(220, 200, n_rows)).astype(int)
    campaign = rng.integers(1, 12, n_rows)
    pdays = rng.choice([-1] * 7 + list(range(1, 400)), n_rows)
    previous = rng.integers(0, 6, n_rows)
    poutcome = rng.choice(POUTCOME, n_rows, p=[0.07, 0.11, 0.05, 0.77])
    do_not_disturb = rng.choice([0, 1], n_rows, p=[0.92, 0.08])

    # Build a target 'y' (did they convert / should this call be prioritized)
    # with realistic, learnable signal baked in — not pure noise — so the
    # trained model produces meaningful, non-random priority scores.
    score = (
        0.010 * (balance / 1000)
        + 0.35 * (poutcome == "success")
        - 0.20 * (poutcome == "failure")
        + 0.015 * duration / 60
        - 0.06 * campaign
        + 0.08 * (previous > 0)
        + 0.05 * (housing == "no")
        - 0.05 * (loan == "yes")
        + 0.03 * (education == "tertiary")
        - 0.10 * do_not_disturb
        + rng.normal(0, 0.35, n_rows)
    )
    prob = 1 / (1 + np.exp(-score))
    y = (rng.random(n_rows) < prob).astype(int)
    y = np.where(y == 1, "yes", "no")

    df = pd.DataFrame({
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "balance": balance, "housing": housing,
        "loan": loan, "contact": contact, "day": day, "month": month,
        "duration": duration, "campaign": campaign, "pdays": pdays,
        "previous": previous, "poutcome": poutcome,
        "do_not_disturb": do_not_disturb, "y": y,
    })
    return df


def main():
    out_path = Path(__file__).resolve().parent / "data" / "data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(out_path, sep=";", index=False)
    print(f"Synthetic dataset written to: {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
