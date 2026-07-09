"""
FastAPI backend for the Intelligent Call Prioritization System.

Run the whole project (data -> preprocessing -> training -> server) with:

    python run.py

Or, if models are already trained, just:

    python -m uvicorn app:app --host 0.0.0.0 --port 8010
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

app = FastAPI(title="Intelligent Call Prioritization System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# The training pipeline always writes a canonical `models/best_model.pkl`
# (whichever algorithm actually won that run), so the API never has to guess
# a hardcoded algorithm name. We still fall back to the older `XGBoost.pkl`
# path for backwards compatibility with earlier runs of this project.
def _load_model():
    candidates = [MODELS_DIR / "best_model.pkl", MODELS_DIR / "XGBoost.pkl"]
    for path in candidates:
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                continue
    return None


def _load_scaler():
    path = MODELS_DIR / "scaler.pkl"
    return joblib.load(path) if path.exists() else None


def _load_json(name: str):
    path = MODELS_DIR / name
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_feature_columns() -> Optional[list[str]]:
    path = MODELS_DIR / "feature_columns.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            cols = json.load(f)
        return [str(c) for c in cols] if isinstance(cols, list) else None
    except Exception:
        return None


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").strip().lower())


def _looks_like_training_schema(columns: list[str]) -> bool:
    normalized = {_normalize_column_name(c) for c in columns}
    required = {
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "donotdisturb",
    }
    return len(required & normalized) >= 8


def _prepare_model_features(df: pd.DataFrame, feature_columns: list[str]):
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]

    if "y" in work.columns:
        work = work.drop(columns=["y"])

    work = work.replace("unknown", pd.NA)

    for col in work.select_dtypes(include="object").columns:
        mode = work[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "unknown"
        work[col] = work[col].fillna(fill_value).astype(str).str.strip()

    for col in work.select_dtypes(include=["int64", "float64"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if {"campaign", "previous"}.issubset(work.columns):
        work["total_contacts"] = work["campaign"] + work["previous"]
        work["contacted_before"] = (work["previous"] > 0).astype(int)
        work["engagement_score"] = work["previous"] * work["campaign"]

    if "poutcome" in work.columns:
        work["previous_success"] = (work["poutcome"].astype(str).str.lower() == "success").astype(int)

    if {"campaign", "pdays"}.issubset(work.columns):
        denom = work["pdays"].where(work["pdays"] >= 0, 0) + 1
        work["contact_intensity"] = work["campaign"] / denom

    encoded = pd.get_dummies(work, drop_first=True)
    bool_cols = encoded.select_dtypes(include="bool").columns
    if len(bool_cols):
        encoded[bool_cols] = encoded[bool_cols].astype(int)

    for col in encoded.columns:
        encoded[col] = pd.to_numeric(encoded[col], errors="coerce")

    aligned = pd.DataFrame(index=encoded.index, columns=feature_columns, dtype=float)
    for feature in feature_columns:
        if feature in encoded.columns:
            aligned[feature] = encoded[feature].fillna(0.0)
        else:
            aligned[feature] = 0.0

    return aligned.astype(float)


def _score_with_trained_model(df: pd.DataFrame):
    if model is None:
        return None, "rule"

    feature_columns = _load_feature_columns()
    if not feature_columns or not _looks_like_training_schema(list(df.columns)):
        return None, "rule"

    try:
        prepared = _prepare_model_features(df, feature_columns)
        if scaler is None:
            return None, "rule"
        scaled = scaler.transform(prepared)
        probs = model.predict_proba(scaled)[:, 1]
        return probs, "model"
    except Exception:
        return None, "rule"


def _detect_delimiter(raw_text: str) -> str:
    sample = raw_text[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        if ";" in sample:
            return ";"
        if "\t" in sample:
            return "\t"
        if "|" in sample:
            return "|"
        return ","


def _load_uploaded_dataframe(upload_file: UploadFile) -> tuple[pd.DataFrame, str]:
    if upload_file.filename is None or not upload_file.filename.lower().endswith(".csv"):
        raise ValueError("Please upload a .csv file.")

    raw_bytes = upload_file.file.read()
    upload_file.file.seek(0)
    if not raw_bytes:
        raise ValueError("The uploaded file is empty.")

    text = None
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise ValueError("Could not decode the CSV file. Please save it as UTF-8.")

    delimiter = _detect_delimiter(text)
    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter, engine="python")
    except Exception as exc:
        raise ValueError(f"Could not parse the CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV has no rows.")

    return df, delimiter


model = _load_model()
scaler = _load_scaler()


class CallIn(BaseModel):
    callerName: Optional[str] = None
    phone: Optional[str] = None
    category: str = "customer_support"
    severity: int = 3
    vulnerable: str = "no"
    vip: str = "no"
    waitMins: int = 0
    location: Optional[str] = None
    summary: Optional[str] = None


class ResolveCallIn(BaseModel):
    callerId: Optional[str] = None
    name: Optional[str] = None
    problem: str = ""
    problemType: str = "other"
    severity: int = 3
    waitMins: int = 0
    vulnerable: str = "no"
    vip: str = "no"


# ---------------------------------------------------------------------------
# Baseline (rule-based) scoring — used for /score, /predict, /resolve-call and
# CSV uploads. This never depends on the ML model, so it always works even
# before a model has been trained, and it can score ANY caller row a user
# throws at it regardless of the ML pipeline's exact training schema.
# ---------------------------------------------------------------------------
def _baseline_score(call: CallIn) -> int:
    sev = max(1, min(int(call.severity), 5))
    wait_mins = max(int(call.waitMins), 0)

    sev_points = sev * 12
    wait_points = min(18, round(wait_mins * 1.2))
    vulnerable_points = 14 if call.vulnerable == "yes" else 0
    vip_points = 10 if call.vip == "yes" else 0

    return min(100, sev_points + wait_points + vulnerable_points + vip_points)


def _bucket_from_score(score: int) -> str:
    if score >= 80:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def _category_for_problem_type(problem_type: str) -> str:
    pt = (problem_type or "other").lower().strip()
    if pt == "billing":
        return "billing"
    if pt == "technical":
        return "technical"
    return "customer_support"


def _resolution_suggestions(
    call: CallIn, problem: str, problem_type: str, priority: str
) -> list[str]:
    p = (problem or "").lower()
    pt = (problem_type or "other").lower().strip()
    steps: list[str] = []

    if priority == "High":
        steps.append(
            "Treat as urgent: resolve on first contact if possible, or escalate to a senior agent within 15 minutes."
        )
    elif priority == "Medium":
        steps.append(
            "Give a clear owner and deadline (e.g. 24 hours); send a written recap after the call."
        )
    else:
        steps.append(
            "Start with the knowledge base and standard runbooks before offering callbacks."
        )

    if pt == "billing" or any(
        k in p for k in ("bill", "charge", "refund", "payment", "invoice")
    ):
        steps.append(
            "Verify identity (account ID, callback number) before discussing balances or refunds."
        )
        steps.append(
            "Pull billing history; walk through line items and the formal dispute path if needed."
        )
    if pt == "technical" or any(
        k in p for k in ("error", "login", "password", "app", "crash", "slow", "bug")
    ):
        steps.append(
            "Capture exact error text, app or browser version, and device type for engineering."
        )
        steps.append(
            "Try safe tier-1 steps (cache clear, reinstall, password reset) from your technical runbook."
        )
    if pt == "complaint":
        steps.append(
            "Acknowledge feelings without debating; focus on concrete fixes you can offer today."
        )
        steps.append(
            "Log a complaint ticket with a verbatim summary and the remedy you proposed."
        )
    if call.vulnerable == "yes":
        steps.append(
            "Vulnerable-caller flag: use plain language, confirm understanding, offer a warm transfer to specialists if required."
        )
    if call.vip == "yes":
        steps.append("VIP account: apply expedited handling and note tier in the CRM.")

    if len((problem or "").strip()) < 8:
        steps.append(
            "Ask what \u201cresolved\u201d looks like for them (refund, replacement, information, or escalation)."
        )

    seen: set[str] = set()
    out: list[str] = []
    for s in steps:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def get_priority(prob: float) -> tuple[int, str]:
    score = int(prob * 100)
    if score >= 80:
        return score, "High"
    if score >= 50:
        return score, "Medium"
    return score, "Low"


# ---------------------------------------------------------------------------
# Static pages
# ---------------------------------------------------------------------------
@app.get("/")
def serve_frontend():
    return FileResponse(str(BASE_DIR / "appp.html"))


@app.get("/health")
def health():
    return {
        "ok": True,
        "modelLoaded": model is not None,
        "scalerLoaded": scaler is not None,
        "processedDataAvailable": (DATA_DIR / "X_test.csv").exists(),
    }


# ---------------------------------------------------------------------------
# Baseline scoring endpoints
# ---------------------------------------------------------------------------
@app.post("/score")
def score(call: CallIn):
    score_value = _baseline_score(call)
    return {"score": score_value}


@app.post("/predict")
def predict(call: CallIn):
    score_value = _baseline_score(call)
    return {"score": score_value, "priority": _bucket_from_score(score_value)}


def _build_resolve_response(body: ResolveCallIn) -> dict:
    call = CallIn(
        callerName=body.name,
        phone=body.callerId,
        category=_category_for_problem_type(body.problemType),
        severity=body.severity,
        vulnerable=body.vulnerable,
        vip=body.vip,
        waitMins=body.waitMins,
        summary=body.problem,
    )
    score_value = _baseline_score(call)
    priority = _bucket_from_score(score_value)
    actions = _resolution_suggestions(call, body.problem, body.problemType, priority)

    cid = (body.callerId or "").strip() or "\u2014"
    nm = (body.name or "").strip() or "Caller"
    log_template = (
        f"Resolution log \u2014 {nm} ({cid})\n"
        f"Issue type: {body.problemType}\n"
        f"Summary: {(body.problem or '').strip()}\n"
        f"Handling priority: {priority} (score {score_value})\n"
        f"Actions taken:\n- \n"
        f"Outcome / next step:\n"
    )

    return {
        "score": score_value,
        "priority": priority,
        "suggested_actions": actions,
        "resolution_log_template": log_template,
    }


@app.post("/resolve-call")
def resolve_call(body: ResolveCallIn):
    return _build_resolve_response(body)


@app.post("/api/resolve-call")
def resolve_call_api_alias(body: ResolveCallIn):
    return _build_resolve_response(body)


# ---------------------------------------------------------------------------
# ML-backed endpoints (trained model over the held-out processed test split)
# ---------------------------------------------------------------------------
def _predict_proba_on_processed(df: pd.DataFrame):
    """
    IMPORTANT: data/processed/X_test.csv is already the *scaled* output of
    the preprocessing pipeline (see data_preprocessing.py -> scale_data).
    Scaling it again here with `scaler.transform(df)` would silently double
    -scale every feature and produce meaningless probabilities. We feed the
    already-scaled frame straight into the model.
    """
    return model.predict_proba(df)[:, 1]


@app.get("/metrics")
def metrics():
    if model is None:
        return {"modelLoaded": False}

    test_path = DATA_DIR / "X_test.csv"
    if not test_path.exists():
        return {"modelLoaded": True, "processedDataAvailable": False}

    df = pd.read_csv(test_path)
    probs = _predict_proba_on_processed(df)

    scores = (probs * 100).astype(int)
    buckets = ["High" if s >= 80 else "Medium" if s >= 50 else "Low" for s in scores]

    total = int(len(scores))
    return {
        "modelLoaded": True,
        "processedDataAvailable": True,
        "total": total,
        "high": int(sum(1 for b in buckets if b == "High")),
        "medium": int(sum(1 for b in buckets if b == "Medium")),
        "low": int(sum(1 for b in buckets if b == "Low")),
        "avgProbability": round(float(probs.mean()), 4) if total else 0,
    }


@app.get("/model-info")
def model_info():
    """Everything the Insights tab needs about the trained model, straight
    from the artifacts model_train.py writes out — no recomputation."""
    metrics_json = _load_json("metrics.json")
    importance_json = _load_json("feature_importance.json")
    return {
        "modelLoaded": model is not None,
        "modelType": type(model).__name__ if model is not None else None,
        "trainingSummary": metrics_json,
        "featureImportance": importance_json,
    }


@app.get("/call-list")
def generate_call_list(limit: int = 20, bucket: Optional[str] = None):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not found. Run `python run.py` once to train it, then restart the server.",
        )

    test_path = DATA_DIR / "X_test.csv"
    if not test_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Processed data not found. Run `python run.py` to build data/processed/.",
        )

    df = pd.read_csv(test_path)
    probs = _predict_proba_on_processed(df)

    df = df.copy()
    df["probability"] = probs

    scores, buckets = [], []
    for p in probs:
        s, b = get_priority(p)
        scores.append(s)
        buckets.append(b)

    df["priority_score"] = scores
    df["priority_bucket"] = buckets
    df = df.sort_values(by="priority_score", ascending=False)

    if bucket and bucket.lower() in ("high", "medium", "low"):
        df = df[df["priority_bucket"].str.lower() == bucket.lower()]

    limit = max(1, min(int(limit), len(df) if len(df) else 1))
    top = df.head(limit).reset_index(drop=True).copy()
    if "name" not in top.columns:
        top["name"] = [f"Customer {k + 1}" for k in range(len(top))]
    if "caller_id" not in top.columns:
        top["caller_id"] = [f"CID-{42000 + k:04d}" for k in range(len(top))]

    return top.to_dict(orient="records")


@app.post("/upload-score")
async def upload_score(file: UploadFile = File(...)):
    """
    Score an arbitrary user-supplied CSV of callers using the rule-based
    baseline engine (works with any column naming, no ML schema required).
    Accepted columns (case-insensitive, punctuation-insensitive, best-effort
    matched): name, caller_id/phone, severity, wait_mins, vulnerable, vip,
    category.
    """
    try:
        df, delimiter = _load_uploaded_dataframe(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalized_columns = {_normalize_column_name(c): c for c in df.columns}

    def col(*names):
        for name in names:
            key = _normalize_column_name(name)
            if key in normalized_columns:
                return normalized_columns[key]
        return None

    name_col = col("name", "customer", "caller_name", "full_name", "caller")
    id_col = col("caller_id", "callerid", "phone", "phone_number", "mobile", "id", "contactid")
    sev_col = col("severity", "priority", "urgency", "scorepriority")
    wait_col = col("wait_mins", "waitmins", "wait_time", "wait", "waitminutes")
    vuln_col = col("vulnerable", "isvulnerable", "vulnerability")
    vip_col = col("vip", "isvip", "premium")
    cat_col = col("category", "type", "issue_type", "problemtype")

    def to_int(v, default=0):
        try:
            return int(float(v))
        except Exception:
            return default

    def to_flag(v):
        s = str(v).strip().lower()
        return "yes" if s in ("yes", "y", "true", "1", "t") else "no"

    probs, scoring_mode = _score_with_trained_model(df)
    results = []
    for i, row in df.iterrows():
        if probs is not None and i < len(probs):
            probability = float(probs[i])
            score_value = int(probability * 100)
            priority = _bucket_from_score(score_value)
            final_score = float(score_value)
            if "balance" in df.columns:
                try:
                    final_score += float(row["balance"] or 0) / 1000.0
                except Exception:
                    pass
            if "poutcome" in df.columns:
                try:
                    final_score += 10.0 if str(row["poutcome"]).strip().lower() == "success" else 0.0
                except Exception:
                    pass
        else:
            call = CallIn(
                callerName=str(row[name_col]) if name_col else None,
                phone=str(row[id_col]) if id_col else None,
                category=str(row[cat_col]) if cat_col else "customer_support",
                severity=to_int(row[sev_col], 3) if sev_col else 3,
                waitMins=to_int(row[wait_col], 0) if wait_col else 0,
                vulnerable=to_flag(row[vuln_col]) if vuln_col else "no",
                vip=to_flag(row[vip_col]) if vip_col else "no",
            )
            score_value = _baseline_score(call)
            priority = _bucket_from_score(score_value)
            final_score = float(score_value)

        results.append({
            "name": str(row[name_col]) if name_col and pd.notna(row[name_col]) else f"Contact {i + 1}",
            "caller_id": str(row[id_col]) if id_col and pd.notna(row[id_col]) else f"CID-{50000 + i}",
            "priority_score": score_value,
            "priority_bucket": priority,
            "final_score": round(final_score, 2),
            "category": str(row[cat_col]) if cat_col and pd.notna(row[cat_col]) else "customer_support",
            "severity": int(row[sev_col]) if sev_col and pd.notna(row[sev_col]) else 3,
            "waitMins": int(row[wait_col]) if wait_col and pd.notna(row[wait_col]) else 0,
            "vulnerable": to_flag(row[vuln_col]) if vuln_col else "no",
            "vip": to_flag(row[vip_col]) if vip_col else "no",
        })

    results.sort(key=lambda r: r["final_score"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    return {"count": len(results), "rows": results, "detected_delimiter": delimiter, "scoring_mode": scoring_mode}


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8010"))
    print()
    print(f"  Open:  http://127.0.0.1:{port}/")
    print(f"  Ctrl+C to stop.")
    print()
    uvicorn.run("app:app", host=host, port=port, reload=False)
