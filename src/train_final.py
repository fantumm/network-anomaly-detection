"""
Trains the final chosen model (soft-voting ensemble: RF + LogisticRegression
+ XGBoost) on the full training set and saves everything needed to run
inference on new data later: the model, the fitted scaler, the label
encoder, the final feature column list, and which columns are numeric.

Run this once you're happy with the model. Re-run it whenever you change
the pipeline (new features, different SMOTE cap, etc.) -- the saved
artifacts must always come from the same preprocessing code as predict.py.
"""
import joblib

from preprocessing import load_and_preprocess
from compare_models import apply_capped_smote
from ensemble_model import build_ensemble
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = "models"


def main():
    print("Loading and preprocessing data...")
    data = load_and_preprocess()
    X_train, y_train = data["X_train"], data["y_train_multi"]

    print("Applying capped SMOTE...")
    X_train_sm, y_train_sm = apply_capped_smote(X_train, y_train)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_sm)

    print("Training final ensemble (RF + LogisticRegression + XGBoost)... this will take a while.")
    ensemble = build_ensemble()
    ensemble.fit(X_train_sm, y_train_enc)

    print(f"Saving artifacts to {MODEL_DIR}/ ...")
    joblib.dump(ensemble, f"{MODEL_DIR}/ensemble_model.joblib")
    joblib.dump(data["scaler"], f"{MODEL_DIR}/scaler.joblib")
    joblib.dump(le, f"{MODEL_DIR}/label_encoder.joblib")
    joblib.dump(data["feature_columns"], f"{MODEL_DIR}/feature_columns.joblib")
    joblib.dump(data["numeric_cols"], f"{MODEL_DIR}/numeric_cols.joblib")

    print("Done. Saved: ensemble_model.joblib, scaler.joblib, label_encoder.joblib, "
          "feature_columns.joblib, numeric_cols.joblib")


if __name__ == "__main__":
    main()