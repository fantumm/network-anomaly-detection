"""
Loads the saved ensemble and runs predictions on new, raw NSL-KDD-format
rows. Handles the exact same encode -> align -> scale pipeline used in
training, so a new row is guaranteed to be processed identically.

Usage:
    from predict import predict_dataframe, predict_single
    result = predict_single({"duration": 0, "protocol_type": "tcp", ...})

    # or, for a CSV of raw rows (same 41 KDD columns, no header, no label col):
    python predict.py path/to/new_traffic.csv
"""
import sys
import joblib
import pandas as pd

from constants import COLUMNS, CATEGORICAL_COLS

MODEL_DIR = "models"
RAW_FEATURE_COLUMNS = COLUMNS[:-1]  # everything except "label"


def _load_artifacts():
    ensemble = joblib.load(f"{MODEL_DIR}/ensemble_model.joblib")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
    label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.joblib")
    feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.joblib")
    numeric_cols = joblib.load(f"{MODEL_DIR}/numeric_cols.joblib")
    return ensemble, scaler, label_encoder, feature_columns, numeric_cols


def _preprocess_new_rows(raw_df, scaler, feature_columns, numeric_cols):
    # One-hot encode the same categorical columns used in training
    encoded = pd.get_dummies(raw_df, columns=CATEGORICAL_COLS)

    # Align to the exact training-time column set: any category not seen
    # during training gets dropped, any training column missing here (e.g.
    # a service type that just doesn't appear in this batch) gets filled
    # with 0. This is the same alignment logic training used, just against
    # a fixed reference list instead of another live DataFrame.
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    # Scale with the *fitted* training scaler -- never re-fit at inference time
    encoded[numeric_cols] = scaler.transform(encoded[numeric_cols])

    return encoded


def predict_dataframe(raw_df):
    """raw_df: DataFrame with the 41 raw KDD feature columns (no label column)."""
    ensemble, scaler, label_encoder, feature_columns, numeric_cols = _load_artifacts()
    X = _preprocess_new_rows(raw_df, scaler, feature_columns, numeric_cols)

    pred_enc = ensemble.predict(X)
    pred_proba = ensemble.predict_proba(X)
    pred_labels = label_encoder.inverse_transform(pred_enc)
    confidences = pred_proba.max(axis=1)

    return pd.DataFrame({
        "prediction": pred_labels,
        "confidence": confidences,
    })


def predict_single(raw_record: dict):
    """raw_record: dict with the 41 raw KDD feature names as keys."""
    raw_df = pd.DataFrame([raw_record])[RAW_FEATURE_COLUMNS]
    result = predict_dataframe(raw_df)
    return result.iloc[0].to_dict()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/new_traffic.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Peek at the raw column count before assigning names, so a labeled
    # test-set CSV (41 features + label) and genuinely new unlabeled data
    # (41 features only) both load correctly instead of silently
    # misaligning columns.
    peek = pd.read_csv(csv_path, header=None, nrows=1)
    n_cols = peek.shape[1]

    if n_cols == len(RAW_FEATURE_COLUMNS):
        raw_df = pd.read_csv(csv_path, header=None, names=RAW_FEATURE_COLUMNS)
    elif n_cols == len(COLUMNS):
        print(f"Note: input has {n_cols} columns (expected {len(RAW_FEATURE_COLUMNS)} raw features) "
              "-- treating the extra trailing column as a label and dropping it for prediction.")
        raw_df = pd.read_csv(csv_path, header=None, names=COLUMNS)
        raw_df = raw_df[RAW_FEATURE_COLUMNS]
    else:
        print(f"Error: input has {n_cols} columns, expected {len(RAW_FEATURE_COLUMNS)} "
              f"(raw features) or {len(COLUMNS)} (raw features + label). Check the file.")
        sys.exit(1)

    results = predict_dataframe(raw_df)
    print(results.to_string())

    print("\nPrediction summary:")
    print(results["prediction"].value_counts())