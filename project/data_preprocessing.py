import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from pathlib import Path
import joblib

# =========================
# 1. Load Dataset
# =========================
def load_data(path: str):
    df = pd.read_csv(path, sep=';')
    df.columns = [c.strip() for c in df.columns]
    return df


# =========================
# 2. Handle Missing Values
# =========================
def handle_missing(df):
    df = df.copy()
    df = df.replace("unknown", np.nan)

    # Fill categorical
    for col in df.select_dtypes(include='object').columns:
        mode = df[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "unknown"
        df[col] = df[col].fillna(fill_value)

    # Fill numeric
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].median())

    # Safety: remove infinities that can appear after feature engineering
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].median())

    return df


# =========================
# 3. Feature Engineering
# (Customer Interaction History)
# =========================
def create_interaction_features(df):

    # Total interactions
    df['total_contacts'] = df['campaign'] + df['previous']

    # Was contacted before
    df['contacted_before'] = df['previous'].apply(lambda x: 1 if x > 0 else 0)

    # Success rate of past campaign
    df['previous_success'] = df['poutcome'].apply(
        lambda x: 1 if x == 'success' else 0
    )

    # Contact frequency (how aggressive campaign is)
    # In this dataset pdays can be -1 ("never contacted"); avoid division by zero.
    denom = df['pdays'].where(df['pdays'] >= 0, 0) + 1
    df['contact_intensity'] = df['campaign'] / denom

    # Customer engagement score (custom feature)
    df['engagement_score'] = df['previous'] * df['campaign']

    return df


# =========================
# 4. Encode Categorical Variables
# =========================
def encode_data(df):

    # Convert target variable first
    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    # One-hot encoding for categorical features
    df = pd.get_dummies(df, drop_first=True)

    # pandas >= 2.0 emits bool dtype columns from get_dummies; cast to int
    # so every downstream consumer (StandardScaler, SMOTE, XGBoost) sees a
    # uniform numeric dtype instead of a silent bool/float mix.
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


# =========================
# 5. Split Features & Target
# =========================
def split_data(df):
    X = df.drop('y', axis=1)
    y = df['y']
    return X, y


# =========================
# 6. Train-Test Split
# =========================
def split_train_test(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


# =========================
# 7. Feature Scaling
# =========================
def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Preserve feature names for downstream saving/training
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train, X_test, scaler


# =========================
# 8. Handle Class Imbalance (SMOTE)
# =========================
def handle_imbalance(X_train, y_train):
    smote = SMOTE(random_state=42)

    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, y_train_res


def save_processed_splits(X_train, X_test, y_train, y_test, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure consistent pandas types for saving
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    y_train = pd.Series(y_train, name="y")
    y_test = pd.Series(y_test, name="y")

    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)


# =========================
# 9. Full Pipeline
# =========================
def preprocessing_pipeline(path: str):

    print("Loading data...")
    df = load_data(path)

    print("Handling missing values...")
    df = handle_missing(df)

    print("Creating interaction features...")
    df = create_interaction_features(df)

    print("Encoding categorical variables...")
    df = encode_data(df)

    print("Splitting data...")
    X, y = split_data(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print("Scaling features...")
    X_train, X_test, scaler = scale_data(X_train, X_test)

    print("Handling class imbalance (SMOTE)...")
    X_train, y_train = handle_imbalance(X_train, y_train)

    print("Preprocessing Completed!")

    return X_train, X_test, y_train, y_test, scaler


# =========================
# 10. Run Script
# =========================
if __name__ == "__main__":
    import json

    default_csv = Path(__file__).resolve().parent / "data" / "data.csv"
    X_train, X_test, y_train, y_test, scaler = preprocessing_pipeline(str(default_csv))

    processed_dir = Path(__file__).resolve().parent / "data" / "processed"
    save_processed_splits(X_train, X_test, y_train, y_test, processed_dir)
    print(f"Saved processed splits to: {processed_dir}")

    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / "scaler.pkl")
    print(f"Saved scaler to: {models_dir / 'scaler.pkl'}")

    # Persist the exact post-encoding feature column order. The API and any
    # future inference code need this to build a compatible row when scoring
    # new, raw data (one-hot columns depend on which categories were seen at
    # training time).
    with open(models_dir / "feature_columns.json", "w") as f:
        json.dump(list(X_train.columns), f)
    print(f"Saved feature column order to: {models_dir / 'feature_columns.json'}")

    print("Final Shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)