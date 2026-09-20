import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

from preprocessing import load_and_preprocess

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed -- run `pip install xgboost` to include it in this comparison.")


def apply_capped_smote(X_train, y_train, cap=10000, k_neighbors=5, random_state=42):
    class_counts = y_train.value_counts()
    smallest_class = class_counts.min()
    safe_k = max(1, min(k_neighbors, smallest_class - 1))
    sampling_strategy = {cls: cap for cls, count in class_counts.items() if count < cap}
    if not sampling_strategy:
        return X_train, y_train
    smote = SMOTE(random_state=random_state, k_neighbors=safe_k, sampling_strategy=sampling_strategy)
    return smote.fit_resample(X_train, y_train)


def main():
    data = load_and_preprocess()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train_multi"], data["y_test_multi"]

    print("Applying capped SMOTE (same as train_model.py, cap=10000)...")
    X_train_sm, y_train_sm = apply_capped_smote(X_train, y_train)
    print(y_train_sm.value_counts())

    # --- Logistic Regression baseline: fast, tells us if there's a simple linear signal ---
    print("\n=== Logistic Regression ===")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    lr.fit(X_train_sm, y_train_sm)
    y_pred_lr = lr.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print(classification_report(y_test, y_pred_lr, zero_division=0))

    # --- XGBoost: boosting focuses iteratively on hard/misclassified examples,
    # often does better than RF on rare, hard-to-separate classes ---
    if HAS_XGB:
        print("\n=== XGBoost ===")
        # XGBoost wants integer-encoded labels
        classes = sorted(y_train_sm.unique())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx_to_class = {i: c for c, i in class_to_idx.items()}
        y_train_enc = y_train_sm.map(class_to_idx)
        y_test_enc = y_test.map(class_to_idx)

        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softmax",
            num_class=len(classes),
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        xgb.fit(X_train_sm, y_train_enc)
        y_pred_xgb_enc = xgb.predict(X_test)
        y_pred_xgb = pd.Series(y_pred_xgb_enc).map(idx_to_class)

        print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
        print(classification_report(y_test, y_pred_xgb, zero_division=0))


if __name__ == "__main__":
    main()