"""
ONE COMMAND TO RUN EVERYTHING.

    python run.py

What it does, in order (each step is skipped automatically if its output
already exists, so re-running this is always safe and fast):

  1. Makes sure required packages are importable (installs any that are
     missing from requirements.txt).
  2. Generates a synthetic training dataset at data/data.csv if you don't
     have a real one yet.
  3. Runs the preprocessing pipeline to build data/processed/*.csv and
     models/scaler.pkl.
  4. Trains Logistic Regression, Random Forest and XGBoost, compares them
     by ROC-AUC, and saves the winner as models/best_model.pkl.
  5. Starts the FastAPI server and opens the dashboard in your browser.

If you already have your own data/data.csv (semicolon-separated, same
columns as the bank-marketing style schema), drop it in before running this
and the generator step will be skipped automatically.

To force a full rebuild (e.g. after changing the dataset), delete the
`data/processed/` and `models/` folders and run this again.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import webbrowser
from pathlib import Path
from threading import Timer

BASE_DIR = Path(__file__).resolve().parent

REQUIRED_PACKAGES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("imblearn", "imbalanced-learn"),
    ("xgboost", "xgboost"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("joblib", "joblib"),
    ("multipart", "python-multipart"),
]


def step(msg: str):
    print(f"\n\033[1;36m==>\033[0m {msg}")


def ensure_packages():
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pip_name)

    if not missing:
        step("All required packages are already installed.")
        return

    step(f"Installing missing packages: {', '.join(missing)}")
    cmd = [sys.executable, "-m", "pip", "install", "-q", *missing]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            "\nCould not auto-install dependencies. Please run:\n"
            f"  {sys.executable} -m pip install -r requirements.txt\n"
            "then run `python run.py` again."
        )
        sys.exit(1)


def ensure_dataset():
    data_csv = BASE_DIR / "data" / "data.csv"
    if data_csv.exists():
        step(f"Using existing dataset: {data_csv}")
        return
    step("No dataset found — generating a synthetic sample dataset...")
    subprocess.run([sys.executable, str(BASE_DIR / "generate_sample_data.py")], check=True)


def ensure_processed():
    processed_dir = BASE_DIR / "data" / "processed"
    needed = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    if processed_dir.exists() and all((processed_dir / f).exists() for f in needed):
        step("Processed data already exists — skipping preprocessing.")
        return
    step("Running preprocessing pipeline...")
    subprocess.run([sys.executable, str(BASE_DIR / "data_preprocessing.py")], check=True)


def ensure_model():
    models_dir = BASE_DIR / "models"
    best_model = models_dir / "best_model.pkl"
    if best_model.exists():
        step("Trained model already exists — skipping training.")
        return
    step("Training models (Logistic Regression, Random Forest, XGBoost)...")
    subprocess.run([sys.executable, str(BASE_DIR / "model_train.py")], check=True)


def open_browser(url: str):
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    print("=" * 64)
    print(" Intelligent Call Prioritization System — one-command setup")
    print("=" * 64)

    ensure_packages()
    ensure_dataset()
    ensure_processed()
    ensure_model()

    host = "127.0.0.1"
    port = 8010
    url = f"http://{host}:{port}/"

    step(f"Starting server at {url}")
    print("  Press Ctrl+C to stop.\n")

    Timer(1.5, open_browser, args=[url]).start()

    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
