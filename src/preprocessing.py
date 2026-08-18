import pandas as pd 
from constants import COLUMNS, ATTACK_CATEGORY_MAP, CATEGORICAL_COLS
from sklearn.preprocessing import StandardScaler


# test_df = pd.read_csv("data/NSL_KDD_Test.csv", header = None, names=COLUMNS)
# train_df = pd.read_csv("data/NSL_KDD_Train.csv", header = None, names=COLUMNS)

# # checking r2l and u2r attacks in test/train data and why its underperforming, 

# # unmapped_train = [l for l in train_df["label"].unique() if l not in ATTACK_CATEGORY_MAP]
# # unmapped_test = [l for l in test_df["label"].unique() if l not in ATTACK_CATEGORY_MAP]
# # print("Unmapped in train:", unmapped_train)
# # print("Unmapped in test:", unmapped_test)


# test_df["attack_category"] = test_df["label"].map(ATTACK_CATEGORY_MAP).fillna("unknown")
# print(test_df["attack_category"].value_counts())


# train_df["attack_category"] = train_df["label"].map(ATTACK_CATEGORY_MAP).fillna("unknown")
# print(train_df["attack_category"].value_counts())

# r2l_test = test_df[test_df["attack_category"] == "r2l"]
# print(r2l_test["label"].value_counts())


# seen_in_train = {"ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster"}

# r2l_test = r2l_test.copy()
# r2l_test["seen_during_training"] = r2l_test["label"].isin(seen_in_train)

# print(r2l_test.groupby("seen_during_training")["label"].count())

#---------------------------------------------------------------------------------------------------

# u2r_test = test_df[test_df["attack_category"] == "u2r"].copy()

# seen_in_train_u2r = {"buffer_overflow", "loadmodule", "perl", "rootkit"}
# u2r_test["seen_during_training"] = u2r_test["label"].isin(seen_in_train_u2r)

# print(u2r_test.groupby("seen_during_training")["label"].count())


# print(u2r_test["label"].value_counts())




# R2L : 76% SEEN / 24% UNSEEN  - IMBALANCED PROBLEM
# U2R : 55% SEEN / 45% UNSEEN - nearly half the U2R test attacks are label types the model literally cannot have learned, no matter how you balance training data.

#---------------------------------------------------------------------------------------------------

# now one-hot encoding and making the data frames share the same columns,

# df = pd.read_csv("data/NSL_KDD_Train.csv", header=None, names=COLUMNS)
# print(df.dtypes)


# train_df = pd.read_csv("data/NSL_KDD_Train.csv", header=None, names=COLUMNS)
# test_df = pd.read_csv("data/NSL_KDD_Test.csv", header=None, names=COLUMNS)

# train_encoded = pd.get_dummies(train_df, columns=CATEGORICAL_COLS)
# # # print(train_encoded.shape)
# # # print(train_encoded.columns.tolist()[:20])  

# test_encoded = pd.get_dummies(test_df, columns=CATEGORICAL_COLS)
# # print(test_encoded.shape)

# train_cols = set(train_encoded.columns)
# test_cols = set(test_encoded.columns)

# # print("In train but not test:", train_cols - test_cols)
# # print("In test but not train:", test_cols - train_cols)

# train_aligned, test_aligned = train_encoded.align(test_encoded, join="left", axis=1, fill_value=0)

# print(train_aligned.shape)
# print(test_aligned.shape)

# both data set are now aligned. 
#---------------------------------------------------------------------------------------------------


# Load raw data
train_df = pd.read_csv("data/NSL_KDD_Train.csv", header=None, names=COLUMNS)
test_df = pd.read_csv("data/NSL_KDD_Test.csv", header=None, names=COLUMNS)

# Derive targets from label, while label still exists
train_df["attack_category"] = train_df["label"].map(ATTACK_CATEGORY_MAP).fillna("unknown")
train_df["is_attack"] = (train_df["attack_category"] != "normal").astype(int)

test_df["attack_category"] = test_df["label"].map(ATTACK_CATEGORY_MAP).fillna("unknown")
test_df["is_attack"] = (test_df["attack_category"] != "normal").astype(int)

# Split X and y BEFORE encoding
y_train = train_df["is_attack"].copy()
y_test = test_df["is_attack"].copy()

X_train_raw = train_df.drop(columns=["label", "attack_category", "is_attack"])
X_test_raw = test_df.drop(columns=["label", "attack_category", "is_attack"])

# encoding categorical columns using one-hot encoding
X_train_encoded = pd.get_dummies(X_train_raw, columns=CATEGORICAL_COLS)
X_test_encoded = pd.get_dummies(X_test_raw, columns=CATEGORICAL_COLS)

X_train_aligned, X_test_aligned = X_train_encoded.align(X_test_encoded, join="left", axis=1, fill_value=0)

# print(X_train_aligned.shape, X_test_aligned.shape)

numeric_cols = [c for c in X_train_aligned.columns
                if not c.startswith("protocol_type_")
                and not c.startswith("service_")
                and not c.startswith("flag_")]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_aligned[numeric_cols] = scaler.fit_transform(X_train_aligned[numeric_cols])
X_test_aligned[numeric_cols] = scaler.transform(X_test_aligned[numeric_cols])



print(X_train_aligned[numeric_cols].describe().loc[["mean", "std"]])

