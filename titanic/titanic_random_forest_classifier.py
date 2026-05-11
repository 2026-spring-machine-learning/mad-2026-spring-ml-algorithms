from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "Titanic-Dataset.csv"
NUMERIC_PREDICTORS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
RESPONSE = "Survived"
MODEL_NAME = "Random Forest"


def load_and_clean_data():
    df = pd.read_csv(DATA_FILE)
    print(f"Titanic {MODEL_NAME} Classification")
    print("=" * 42)
    print(f"Raw shape: {df.shape}")
    print("\nNA counts before cleaning:")
    print(df.isna().sum())

    df = df.dropna()
    print(f"\nShape after dropping NA rows: {df.shape}")
    return df


def encode_categorical(df):
    """Binary-encode Sex (2 values) and one-hot encode Embarked (3 values)."""
    df = df.copy()

    # Sex has 2 values → label/binary encode: male=0, female=1
    df["Sex_encoded"] = df["Sex"].map({"male": 0, "female": 1})

    # Embarked has 3 values → one-hot encode; drop first to avoid multicollinearity
    embarked_dummies = (
        pd.get_dummies(df["Embarked"], prefix="Embarked", drop_first=True)
        .astype(int)
    )
    df = pd.concat([df, embarked_dummies], axis=1)

    cat_cols = ["Sex_encoded"] + list(embarked_dummies.columns)
    print(f"\nEncoded columns added: {cat_cols}")
    print(f"  Sex_encoded  → male=0, female=1  | counts: {df['Sex_encoded'].value_counts().to_dict()}")
    for col in embarked_dummies.columns:
        print(f"  {col:<15} | counts: {df[col].value_counts().to_dict()}")

    return df, cat_cols


def select_features(data, predictors):
    print(f"\n{'Predictor':<15} {'Pearson R':>10} {'p-value':>12} {'Keep?':>8}")
    print("-" * 50)

    kept_predictors = []
    for col in predictors:
        r_value, p_value = stats.pearsonr(data[col], data[RESPONSE])
        keep = abs(r_value) >= 0.1
        if keep:
            kept_predictors.append(col)
        print(
            f"{col:<15} {r_value:>10.4f} {p_value:>12.4f} "
            f"{'YES' if keep else 'NO':>8}"
        )

    print(f"\nPredictors retained (|R| >= 0.10): {kept_predictors}")
    print(
        "\nIndependence check: the rows represent different passengers, "
        "so observations are treated as independent."
    )
    return kept_predictors


def split_dataset(data, predictors, random_state):
    X = data[predictors]
    y = data[RESPONSE]
    return train_test_split(X, y, test_size=0.2, random_state=random_state)


def fit_random_forest_model(X_train, y_train, random_state):
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def print_prediction_details(prediction):
    print("\nPrediction array (first 30 values):")
    print(prediction[:30])
    print("\nUnique values in prediction:", np.unique(prediction))
    print("Predictions are all 0s and 1s because this is classification.")


def get_tp_tn(y_test, prediction):
    cm = confusion_matrix(y_test, prediction)
    tp = cm[1][1]
    tn = cm[0][0]
    return tp, tn


def print_model_results(y_test, prediction, random_state, label=""):
    accuracy = accuracy_score(y_test, prediction)
    tp, tn = get_tp_tn(y_test, prediction)
    tag = f" [{label}]" if label else ""
    print(f"\n{MODEL_NAME} Accuracy (random_state={random_state}){tag}: {accuracy:.4f}")
    print(f"  True Positives (TP): {tp}")
    print(f"  True Negatives (TN): {tn}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, prediction))
    print("\nClassification Report:")
    print(classification_report(y_test, prediction))
    return accuracy, tp, tn


def compare_all_ones_baseline(y_test, model_accuracy, random_state, label=""):
    all_ones = np.ones(len(y_test), dtype=int)
    baseline_accuracy = accuracy_score(y_test, all_ones)
    tag = f" [{label}]" if label else ""
    print(f"All-1s Baseline Accuracy (random_state={random_state}){tag}: {baseline_accuracy:.4f}")
    if abs(model_accuracy - baseline_accuracy) < 0.02:
        print(
            f"The all-1s baseline performs about as well as {MODEL_NAME.lower()} "
            "on this split."
        )
    elif model_accuracy > baseline_accuracy:
        print(f"{MODEL_NAME} performs better than the all-1s baseline.")
    else:
        print(f"The all-1s baseline performs better than {MODEL_NAME.lower()}.")
    return baseline_accuracy


def main():
    df = load_and_clean_data()
    df, cat_cols = encode_categorical(df)

    all_predictors = NUMERIC_PREDICTORS + cat_cols
    data = df[all_predictors + [RESPONSE]].copy()
    kept_predictors = select_features(data, all_predictors)

    # ── WITHOUT balancing ─────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("WITHOUT BALANCING")
    print("=" * 50)

    X_train, X_test, y_train, y_test = split_dataset(
        data, kept_predictors, random_state=1
    )
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")

    model = fit_random_forest_model(X_train, y_train, random_state=1)
    prediction = model.predict(X_test)

    print_prediction_details(prediction)
    accuracy, tp_before, tn_before = print_model_results(
        y_test, prediction, random_state=1, label="before balancing"
    )
    compare_all_ones_baseline(y_test, accuracy, random_state=1, label="before balancing")

    # ── WITH RandomOverSampler balancing ──────────────────────────────────
    print("\n" + "=" * 50)
    print("WITH RANDOM OVER-SAMPLING (random_state=1)")
    print("=" * 50)

    ros = RandomOverSampler(random_state=1)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    print(f"\nOriginal training class distribution:\n{pd.Series(y_train).value_counts().to_string()}")
    print(f"\nResampled training samples: {X_train_res.shape[0]}")
    print(f"Resampled class distribution:\n{pd.Series(y_train_res).value_counts().to_string()}")

    model_res = fit_random_forest_model(X_train_res, y_train_res, random_state=1)
    prediction_res = model_res.predict(X_test)

    accuracy_res, tp_after, tn_after = print_model_results(
        y_test, prediction_res, random_state=1, label="after balancing"
    )
    compare_all_ones_baseline(y_test, accuracy_res, random_state=1, label="after balancing")

    # ── TP / TN SUMMARY ───────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("TRUE POSITIVES & TRUE NEGATIVES SUMMARY")
    print("=" * 50)
    print(f"{'':25} {'TP':>6} {'TN':>6}")
    print(f"{'Before balancing':25} {tp_before:>6} {tn_before:>6}")
    print(f"{'After balancing':25}  {tp_after:>6} {tn_after:>6}")


if __name__ == "__main__":
    main()
