import pandas as pd
from sklearn.preprocessing import StandardScaler
from constants import COLUMNS, ATTACK_CATEGORY_MAP, CATEGORICAL_COLS


def load_and_preprocess(train_path="data/NSL_KDD_Train.csv", test_path="data/NSL_KDD_Test.csv"):
    """
    Loads NSL-KDD train/test CSVs, derives the binary is_attack target,
    one-hot encodes categorical columns, aligns train/test columns,
    and scales numeric features.

    Returns: X_train, X_test, y_train, y_test
    """

    # Load raw data
    train_df = pd.read_csv(train_path, header=None, names=COLUMNS)
    test_df = pd.read_csv(test_path, header=None, names=COLUMNS)

    # Derive targets from label, while label still exists
    train_df["attack_category"] = train_df["label"].map(ATTACK_CATEGORY_MAP).fillna("unknown")
    train_df["is_attack"] = (train_df["attack_category"] != "normal").astype(int)

    test_df["attack_category"] = test_df["label"].map(ATTACK_CATEGORY_MAP).fillna("unknown")
    test_df["is_attack"] = (test_df["attack_category"] != "normal").astype(int)

    # Split X and y BEFORE encoding
    y_train_binary = train_df["is_attack"].copy()
    y_test_binary = test_df["is_attack"].copy()

    y_train_multi = train_df["attack_category"].copy()
    y_test_multi = test_df["attack_category"].copy()

    X_train_raw = train_df.drop(columns=["label", "attack_category", "is_attack"])
    X_test_raw = test_df.drop(columns=["label", "attack_category", "is_attack"])

    # One-hot encode categorical columns
    X_train_encoded = pd.get_dummies(X_train_raw, columns=CATEGORICAL_COLS)
    X_test_encoded = pd.get_dummies(X_test_raw, columns=CATEGORICAL_COLS)

    # Align columns (handles categories seen in one split but not the other)
    X_train_aligned, X_test_aligned = X_train_encoded.align(
        X_test_encoded, join="left", axis=1, fill_value=0
    )

    # Scale numeric columns only (leave one-hot dummy columns as 0/1)
    numeric_cols = [
        c for c in X_train_aligned.columns
        if not c.startswith("protocol_type_")
        and not c.startswith("service_")
        and not c.startswith("flag_")
    ]

    scaler = StandardScaler()
    X_train_aligned[numeric_cols] = scaler.fit_transform(X_train_aligned[numeric_cols])
    X_test_aligned[numeric_cols] = scaler.transform(X_test_aligned[numeric_cols])

    return {
        "X_train": X_train_aligned,
        "X_test": X_test_aligned,
        "y_train_binary": y_train_binary,
        "y_test_binary": y_test_binary,
        "y_train_multi": y_train_multi,
        "y_test_multi": y_test_multi,
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "feature_columns": X_train_aligned.columns.tolist(),
    }


if __name__ == "__main__":
    data = load_and_preprocess()
    print("X_train:", data["X_train"].shape, " X_test:", data["X_test"].shape)
    print("y_train_binary distribution:\n", data["y_train_binary"].value_counts(normalize=True))
    print("\ny_train_multi distribution:\n", data["y_train_multi"].value_counts(normalize=True))