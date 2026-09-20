import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

from preprocessing import load_and_preprocess
from compare_models import apply_capped_smote


def build_ensemble():
    """
    Soft-voting ensemble of three models that each cover a different
    weak spot found during experimentation:
      - RandomForest: strong on dos/probe/normal
      - LogisticRegression: much better recall on r2l/u2r
      - XGBoost: strong precision on r2l, decent u2r balance
    Soft voting averages each model's predicted probabilities per class,
    so a class doesn't need to "win" with every model -- it needs the
    highest *combined* confidence.
    """
    rf = RandomForestClassifier(
        n_estimators=200, max_features=None, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1,
    )
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        objective="multi:softprob", n_jobs=-1, eval_metric="mlogloss",
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("lr", lr), ("xgb", xgb)],
        voting="soft",
        weights=[1, 1, 1],  # start equal; tune later based on results
    )
    return ensemble


def main():
    data = load_and_preprocess()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train_multi"], data["y_test_multi"]

    print("Applying capped SMOTE...")
    X_train_sm, y_train_sm = apply_capped_smote(X_train, y_train)

    # VotingClassifier + XGBoost both need numeric labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_sm)
    y_test_enc = le.transform(y_test)

    print("Training ensemble (RF + LogisticRegression + XGBoost)... this will take a while.")
    ensemble = build_ensemble()
    ensemble.fit(X_train_sm, y_train_enc)

    y_pred_enc = ensemble.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    print(f"\n=== Ensemble Accuracy: {accuracy_score(y_test, y_pred):.4f} ===")
    labels = sorted(y_test.unique())
    print("\nConfusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=[f"actual_{l}" for l in labels], columns=[f"pred_{l}" for l in labels]))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))


if __name__ == "__main__":
    main()