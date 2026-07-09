import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
import os

# =========================
# 1. Train Models
# =========================
def train_models(X_train, y_train):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        # NOTE: `use_label_encoder` was removed in xgboost>=2.0 and raises a
        # TypeError if passed. Omitting it keeps this compatible with both
        # old and new xgboost releases.
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


# =========================
# 2. Evaluate Models
# =========================
def evaluate_models(models, X_test, y_test):

    results = []

    for name, model in models.items():

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        print(f"\n{name} Results:")
        print(classification_report(y_test, y_pred))

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": roc
        })

    return pd.DataFrame(results)


# =========================
# 3. Save Best Model
# =========================
def save_best_model(models, results_df, feature_names=None):

    best_model_name = results_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]["Model"]
    best_model = models[best_model_name]

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, f"models/{best_model_name}.pkl")

    # The API loads a single canonical file so it never has to guess which
    # algorithm happened to win this run.
    joblib.dump(best_model, "models/best_model.pkl")

    print(f"\nBest Model: {best_model_name} saved successfully!")

    # Persist metrics + (if available) feature importance so the API/dashboard
    # can show real model insight without retraining or recomputing anything.
    import json

    metrics_payload = {
        "best_model": best_model_name,
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "leaderboard": results_df.round(4).to_dict(orient="records"),
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    importances = None
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])

    if importances is not None and feature_names is not None:
        pairs = sorted(
            zip(feature_names, [float(x) for x in importances]),
            key=lambda kv: kv[1],
            reverse=True,
        )[:15]
        with open("models/feature_importance.json", "w") as f:
            json.dump([{"feature": k, "importance": v} for k, v in pairs], f, indent=2)

    return best_model, best_model_name


# =========================
# 4. Probability-Based Scoring
# =========================
def predict_with_score(model, X_sample):

    prob = model.predict_proba(X_sample)[:, 1]

    # Convert probability into priority score (0–100)
    score = (prob * 100).astype(int)

    result = ["YES" if p > 0.5 else "NO" for p in prob]

    return result, prob, score


# =========================
# 5. Main Function
# =========================
def run_training_pipeline(X_train, X_test, y_train, y_test):

    # Train
    models = train_models(X_train, y_train)

    # Evaluate
    results_df = evaluate_models(models, X_test, y_test)
    print("\nModel Comparison:\n", results_df)

    # Save best
    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None
    best_model, best_name = save_best_model(models, results_df, feature_names=feature_names)

    return best_model, results_df


# =========================
# Example Usage
# =========================
if __name__ == "__main__":

    # Load your preprocessed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    best_model, results = run_training_pipeline(X_train, X_test, y_train, y_test)

    # Example prediction
    sample = X_test.iloc[:5]

    result, prob, score = predict_with_score(best_model, sample)

    print("\nSample Predictions:")
    for i in range(len(result)):
        print(f"Prediction: {result[i]}, Probability: {prob[i]:.2f}, Score: {score[i]}")