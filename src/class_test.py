import pandas as pd 
from constants import COLUMNS, ATTACK_CATEGORY_MAP

test_df = pd.read_csv("/Users/rewant/Documents/cv projects/networka/data/NSL_KDD_Test.csv", header = None, names=COLUMNS)
train_df = pd.read_csv("/Users/rewant/Documents/cv projects/networka/data/NSL_KDD_Train.csv", header = None, names=COLUMNS)

# checking r2l and u2r attacks in test/train data and why its underperforming, 

# unmapped_train = [l for l in train_df["label"].unique() if l not in ATTACK_CATEGORY_MAP]
# unmapped_test = [l for l in test_df["label"].unique() if l not in ATTACK_CATEGORY_MAP]
# print("Unmapped in train:", unmapped_train)
# print("Unmapped in test:", unmapped_test)


test_df["attack_category"] = test_df["label"].map(ATTACK_CATEGORY_MAP).fillna("unknown")
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
