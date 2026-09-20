import pandas as pd
from preprocessing import load_and_preprocess
from TRAIN_final import train_random_forest
from constants import COLUMNS, ATTACK_CATEGORY_MAP

# Content-based features -- these are what r2l/u2r attacks should show up in
# (failed logins, privilege escalation, file/shell access), as opposed to the
# traffic-volume features (bytes, count, service) that dominate dos/probe detection.
CONTENT_FEATURES = [
    "num_failed_logins", "hot", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "is_guest_login", "logged_in",
]


def main():
    data = load_and_preprocess()
    X_train, y_train_multi = data["X_train"], data["y_train_multi"]

    print("Training multi-class RF (with SMOTE) for importance audit...")
    model = train_random_forest(X_train, y_train_multi, use_smote=True)

    all_importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
    ranks = all_importances.rank(ascending=False).astype(int)

    print(f"\n=== Content-based feature ranks (out of {len(all_importances)} total features) ===")
    for feat in CONTENT_FEATURES:
        if feat in all_importances.index:
            print(f"{feat:30s} importance={all_importances[feat]:.6f}  rank={ranks[feat]}")
        else:
            print(f"{feat:30s} NOT FOUND in feature columns")

    # Raw (unscaled) means by attack category -- checks whether real separation
    # exists in the data at all, independent of what the model chose to use.
    print("\n=== Raw content-feature means by attack_category (train set) ===")
    train_df = pd.read_csv("data/NSL_KDD_Train.csv", header=None, names=COLUMNS)
    train_df["attack_category"] = train_df["label"].map(ATTACK_CATEGORY_MAP).fillna("unknown")
    print(train_df.groupby("attack_category")[CONTENT_FEATURES].mean().T)

    # How many r2l/u2r rows actually have non-zero values in these features at all?
    print("\n=== % of rows with non-zero value, by attack_category ===")
    nonzero_pct = train_df.groupby("attack_category")[CONTENT_FEATURES].apply(
        lambda g: (g != 0).mean() * 100
    )
    print(nonzero_pct.T)


if __name__ == "__main__":
    main()