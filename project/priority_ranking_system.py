import pandas as pd
import numpy as np
import joblib
from datetime import datetime, time
from pathlib import Path

# =========================
# 1. Load Model & Scaler
# =========================
def _load_best_model():
    for name in ("best_model.pkl", "XGBoost.pkl"):
        p = Path("models") / name
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError(
        "No trained model found in models/. Run `python run.py` first."
    )


model = _load_best_model()
scaler = joblib.load("models/scaler.pkl")


# =========================
# 2. Convert Probability → Score → Bucket
# =========================
def assign_priority(prob):

    score = int(prob * 100)

    if score >= 80:
        bucket = "High"
    elif score >= 50:
        bucket = "Medium"
    else:
        bucket = "Low"

    return score, bucket


# =========================
# 3. Business Constraints
# =========================
def apply_business_rules(df):

    current_time = datetime.now().time()

    # Rule 1: Do Not Disturb
    if "do_not_disturb" in df.columns:
        df = df[df["do_not_disturb"] == 0]

    # Rule 2: Calling Time (9 AM – 7 PM)
    start_time = time(9, 0)
    end_time = time(19, 0)

    if not (start_time <= current_time <= end_time):
        print("Outside calling hours → No calls allowed")
        return pd.DataFrame()

    # Rule 3: Limit excessive calls
    if "campaign" in df.columns:
        df = df[df["campaign"] < 5]

    return df


# =========================
# 4. Predict Probability
# =========================
def predict_probabilities(df):

    X = df.copy()

    # If you're passing already-scaled features (e.g. data/processed/X_test.csv),
    # don't scale again.
    probs = model.predict_proba(X)[:, 1]

    df["probability"] = probs

    return df


# =========================
# 5. Assign Priority Buckets
# =========================
def assign_priorities(df):

    scores = []
    buckets = []

    for p in df["probability"]:
        score, bucket = assign_priority(p)
        scores.append(score)
        buckets.append(bucket)

    df["priority_score"] = scores
    df["priority_bucket"] = buckets

    return df


# =========================
# 6. Dynamic Ranking
# =========================
def rank_customers(df):

    # Rank based on:
    # 1. Priority score
    # 2. Customer value (balance)
    # 3. Past success

    if "balance" in df.columns:
        df["final_score"] = df["priority_score"] + (df["balance"] / 1000)
    else:
        df["final_score"] = df["priority_score"]

    if "previous_success" in df.columns:
        df["final_score"] += df["previous_success"] * 10

    # Sort descending (highest priority first)
    df = df.sort_values(by="final_score", ascending=False)

    df["rank"] = range(1, len(df) + 1)

    return df


# =========================
# 7. Full Pipeline
# =========================
def run_priority_system(input_path):

    print("Loading data...")
    df = pd.read_csv(input_path)

    print("Applying business constraints...")
    df = apply_business_rules(df)

    if df.empty:
        print("No customers eligible for calling")
        return df

    print("Predicting probabilities...")
    df = predict_probabilities(df)

    print("Assigning priority buckets...")
    df = assign_priorities(df)

    print("Ranking customers dynamically...")
    df = rank_customers(df)

    print("Done!")

    return df


# =========================
# 8. Run Example
# =========================
if __name__ == "__main__":
    result_df = run_priority_system(str(Path("data") / "processed" / "X_test.csv"))

    print("\nTop 10 Customers to Call:")
    print(result_df.head(10)[
        ["probability", "priority_score", "priority_bucket", "final_score", "rank"]
    ])