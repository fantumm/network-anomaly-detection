# networka — Network Anomaly Detector (NSL-KDD)

A network intrusion/anomaly detector trained on the NSL-KDD dataset. Classifies
network connection records as normal traffic or one of four attack categories
(dos, probe, r2l, u2r), using an ensemble of Random Forest, Logistic Regression,
and XGBoost.
---

## Dataset

[NSL-KDD]("https://github.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/tree/master") 
Each row is a network connection record with 41
features (duration, protocol, bytes transferred, login attempts, error rates,
etc.) plus a label naming the specific attack (or `normal`).

Labels are grouped into 5 categories via `ATTACK_CATEGORY_MAP` in `constants.py`:

| Category | Meaning | Example attack types |
|---|---|---|
| `normal` | legitimate traffic | — |
| `dos` | denial of service | neptune, smurf, back |
| `probe` | surveillance/scanning | nmap, portsweep, satan |
| `r2l` | remote-to-local (unauthorized remote access) | guess_passwd, ftp_write, warezclient |
| `u2r` | user-to-root (privilege escalation) | buffer_overflow, rootkit, perl |

**Critical property of this dataset**: the official test set intentionally
contains attack subtypes that never appear in the training set at all. This
was confirmed early in this project:
- **~24% of r2l test-set attacks** are subtypes unseen during training
- **~45% of u2r test-set attacks** are subtypes unseen during training

This isn't a data quality issue — it's the dataset's design. NSL-KDD is built
to penalize models that just memorize known attack signatures rather than
learning generalizable patterns. Keep this in mind when reading every result
below.

---

## Pipeline overview

1. **`preprocessing.py`** — loads train/test CSVs, derives `is_attack`
   (binary) and `attack_category` (multi-class) targets, one-hot encodes
   categorical columns (`protocol_type`, `service`, `flag`), aligns train/test
   columns so both share the same schema, and scales numeric columns with
   `StandardScaler` (fit on train only). Returns a dict with `X_train`,
   `X_test`, both target variants, the fitted `scaler`, `numeric_cols`, and
   `feature_columns` — the last three are needed later for inference.

2. **`train_model.py`** — the original Random Forest experiments. Supports
   capped SMOTE oversampling (`smote_cap`, default 10,000 — deliberately
   *not* oversampling tiny classes like u2r all the way up to majority size,
   since ~52 real samples stretched to 67k copies produces a dense,
   low-diversity synthetic cluster) and `max_features`/`class_weight` tuning.

3. **`feature_audit.py`** — diagnostic script that checks whether
   content-based features (`num_failed_logins`, `hot`, `root_shell`, etc.)
   actually separate r2l/u2r from normal traffic in the raw data, and where
   Random Forest ranks them in its own feature importances. Confirmed real
   signal exists (e.g. `hot` averages 8.3 for r2l vs 0.04 for dos) but RF was
   under-using it.

4. **`compare_models.py`** — trains Logistic Regression and XGBoost on the
   same preprocessed, capped-SMOTE data, to check whether r2l/u2r difficulty
   was specific to Random Forest or a property of the data itself. Logistic
   Regression turned out to recall r2l far better than RF (0.19 vs ~0.01-0.04).

5. **`ensemble_model.py`** — soft-voting ensemble of RandomForest +
   LogisticRegression + XGBoost, blending each model's predicted
   probabilities. This is the final chosen model.

6. **`train_final_model.py`** — fits the ensemble once on the full training
   set and saves it, plus the scaler, label encoder, and column lists,
   to `models/` via `joblib`.

7. **`predict.py`** — loads the saved artifacts and runs inference on new,
   raw connection records (single dict or CSV), applying the exact same
   encode → align → scale steps used in training.

8. **`cross_validate.py`** — 3-fold stratified CV on the training set, with
   SMOTE re-applied fresh inside each fold (no leakage from oversampling).
   See [Known limitations](#known-limitations) for why its numbers look much
   better than the real test-set numbers, and why that's expected.

---

## Results

All numbers below are from the **final ensemble model, evaluated on the
official NSL-KDD test set** (the only trustworthy generalization measure for
this dataset — see limitations).

### Binary (`is_attack`: normal vs. attack)
Not the final target of this project (multi-class is), but tracked alongside it:

| Metric | Value |
|---|---|
| Accuracy | 0.80 |
| Attack precision | 0.97 |
| Attack recall | 0.67 |

### Multi-class (`attack_category`)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| dos | 0.96 | 0.83 | 0.89 | 7,458 |
| normal | 0.70 | 0.97 | 0.81 | 9,711 |
| probe | 0.88 | 0.81 | 0.84 | 2,421 |
| r2l | 0.94 | 0.09 | 0.16 | 2,887 |
| u2r | 0.62 | 0.30 | 0.40 | 67 |

Overall accuracy: **0.79**

### Cross-validation (training set only, 3-fold)
Mean macro F1: **0.93 (± 0.02)** — dramatically higher than the real test-set
macro F1 of ~0.62. This gap is expected and explained below, not a bug.

---

## Known limitations

**1. r2l detection is the project's core unsolved weak point.**
Despite: (a) capped SMOTE, (b) `max_features=None` + `class_weight` tuning,
(c) trying Logistic Regression and XGBoost, and (d) a 3-model ensemble — r2l
recall tops out around 0.09-0.19 depending on model. Root causes identified
during this project:
- r2l bundles structurally different attacks (password guessing, guest-login
  abuse, protocol-level exploits) that don't share one behavioral signature.
- Real signal exists in content-based features (`hot`, `is_guest_login`,
  `num_failed_logins`) — confirmed via `feature_audit.py` — but models
  struggle to weight these consistently against the ~70 `service_*` one-hot
  columns and volume-based traffic features that dominate the feature space.
- A meaningful fraction of test-set r2l/u2r attacks are subtypes never seen
  in training at all (see below) — no amount of modeling fixes this part.

**2. NSL-KDD's test set contains genuinely unseen attack subtypes.**
~24% of r2l and ~45% of u2r test attacks are subtypes absent from training.
This is a hard ceiling for any supervised model trained only on the provided
training labels — it structurally cannot learn signatures it has never seen.
Closing this gap would require either additional labeled data covering those
subtypes, or an unsupervised/semi-supervised approach (e.g. anomaly detection
trained only on normal traffic, catching *any* deviation rather than matching
known attack signatures) — see future work.

**3. Cross-validation overestimates real performance for this dataset.**
CV folds are drawn entirely from the training set, so the model is validated
against attack subtypes it has already learned from repeatedly (just not
those exact rows). This is why CV macro F1 (0.93) is so much higher than the
real test-set macro F1 (~0.62). **Always trust the official test-set numbers
over CV numbers for this project.** CV is still useful here for one thing:
confirming training is stable and not wildly sensitive to which rows land in
the training set (std of only 0.02 across folds says training is stable).

There's also a minor, secondary leakage source: the `StandardScaler` in
`preprocessing.py` is fit once on the full training set before CV splitting,
so each fold's validation rows technically contributed a small amount to
their own feature scaling. This has a much smaller effect than point 3 above,
but is worth knowing about if you extend the CV script.

---

## How to run

```bash
# 1. Set up environment
pip install -r requirements.txt   # or install individually, see below

# 2. Sanity-check preprocessing
python src/preprocessing.py

# 3. (Optional) Run the original Random Forest experiments
python src/train_model.py

# 4. (Optional) Diagnose feature usefulness for r2l/u2r
python src/feature_audit.py

# 5. (Optional) Compare Logistic Regression / XGBoost against RF
python src/compare_models.py

# 6. (Optional) Evaluate the ensemble on the standard train/test split
python src/ensemble_model.py

# 7. Train and persist the final ensemble (required before predict.py)
python src/train_final_model.py

# 8. Run inference
python src/predict.py path/to/new_traffic.csv       # 41 raw feature columns
python src/predict.py data/NSL_KDD_Test.csv          # or a labeled file — label is auto-dropped

# 9. (Optional, slow) Cross-validate for training stability
python src/cross_validate.py
```

Steps 2-6 are optional if you just want the final trained model — they're
the experimentation trail that led to the final ensemble design and are kept
for reproducibility and explanation.

### Dependencies
```
pandas
scikit-learn
imbalanced-learn
xgboost
joblib
```

---

## File structure

```
networka/
├── data/
│   ├── NSL_KDD_Train.csv
│   └── NSL_KDD_Test.csv
├── models/                      # created by train_final_model.py
│   ├── ensemble_model.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   ├── feature_columns.joblib
│   └── numeric_cols.joblib
├── src/
│   ├── constants.py             # COLUMNS, CATEGORICAL_COLS, ATTACK_CATEGORY_MAP
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── feature_audit.py
│   ├── compare_models.py
│   ├── ensemble_model.py
│   ├── train_final_model.py
│   ├── predict.py
│   └── cross_validate.py
└── README.md
```

---

## Possible future work

- **Unsupervised layer for zero-day generalization**: train an Isolation
  Forest or autoencoder on normal traffic only, to flag genuinely novel
  patterns rather than matching known attack signatures — the most direct
  way to address the unseen-subtype ceiling on r2l/u2r.
- **Per-subtype error analysis**: break r2l/u2r predictions down by original
  attack label (not just category) to see which specific subtypes the model
  handles well vs. fails on completely.
- **Feature engineering for sparse content features**: `num_failed_logins`,
  `hot`, etc. are heavily zero-inflated; a log-transform or different scaling
  strategy before `StandardScaler` might preserve their signal better.
- **Reduce `service_*` dimensionality**: ~70 one-hot columns from a single
  categorical feature dilutes the feature space; grouping rare services into
  an "other" bucket could help models allocate attention better.
- **Tune ensemble voting weights**: current weights are equal (`[1,1,1]`);
  weighting Logistic Regression higher specifically for r2l/u2r classes
  (e.g. via a `StackingClassifier` with a learned meta-model) could recover
  some of the recall lost when RF/XGBoost outvote LR under equal weighting.
- **Wrap `predict.py` as an API** (Flask/FastAPI) for real inference use
  rather than CLI/script-only usage.