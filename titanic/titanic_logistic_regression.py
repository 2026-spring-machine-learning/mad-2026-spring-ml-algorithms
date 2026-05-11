from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "Titanic-Dataset.csv"
PREDICTORS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
RESPONSE = "Survived"


def load_and_clean_data():
    df = pd.read_csv(DATA_FILE)
    print("Titanic Logistic Regression Classification")
    print("=" * 42)
    print(f"Raw shape: {df.shape}")
    print("\nNA counts before cleaning:")
    print(df.isna().sum())

    df = df.dropna()
    print(f"\nShape after dropping NA rows: {df.shape}")
    return df


def select_features(data):
    print(f"\n{'Predictor':<10} {'Pearson R':>10} {'p-value':>12} {'Keep?':>8}")
    print("-" * 45)

    kept_predictors = []
    for col in PREDICTORS:
        r_value, p_value = stats.pearsonr(data[col], data[RESPONSE])
        keep = abs(r_value) >= 0.1
        if keep:
            kept_predictors.append(col)
        print(
            f"{col:<10} {r_value:>10.4f} {p_value:>12.4f} "
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


def fit_logistic_model(X_train, y_train, random_state):
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def print_prediction_details(prediction):
    print("\nPrediction array (first 30 values):")
    print(prediction[:30])
    print("\nUnique values in prediction:", np.unique(prediction))
    print("Predictions are all 0s and 1s because this is classification.")


def print_model_results(y_test, prediction, random_state):
    accuracy = accuracy_score(y_test, prediction)
    print(f"\nLogistic Regression Accuracy (random_state={random_state}): {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, prediction))
    print("\nClassification Report:")
    print(classification_report(y_test, prediction))
    return accuracy


def compare_all_ones_baseline(y_test, model_accuracy, random_state):
    all_ones = np.ones(len(y_test), dtype=int)
    baseline_accuracy = accuracy_score(y_test, all_ones)

    print(f"All-1s Baseline Accuracy (random_state={random_state}): {baseline_accuracy:.4f}")
    if abs(model_accuracy - baseline_accuracy) < 0.02:
        print(
            "The all-1s baseline performs about as well as logistic regression "
            "on this split."
        )
    elif model_accuracy > baseline_accuracy:
        print("Logistic regression performs better than the all-1s baseline.")
    else:
        print("The all-1s baseline performs better than logistic regression.")

    return baseline_accuracy


def compare_random_state_zero(data, predictors):
    X_train, X_test, y_train, y_test = split_dataset(data, predictors, random_state=0)
    model = fit_logistic_model(X_train, y_train, random_state=0)
    prediction = model.predict(X_test)
    all_ones = np.ones(len(y_test), dtype=int)

    model_accuracy = accuracy_score(y_test, prediction)
    baseline_accuracy = accuracy_score(y_test, all_ones)

    print("\nrandom_state=0 comparison:")
    print(f"Logistic Regression Accuracy: {model_accuracy:.4f}")
    print(f"All-1s Baseline Accuracy:    {baseline_accuracy:.4f}")
    if baseline_accuracy > model_accuracy:
        print("With random_state=0, the all-1s baseline beats the algorithm.")


def main():
    df = load_and_clean_data()
    data = df[PREDICTORS + [RESPONSE]].copy()
    kept_predictors = select_features(data)

    X_train, X_test, y_train, y_test = split_dataset(
        data, kept_predictors, random_state=1
    )
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")

    model = fit_logistic_model(X_train, y_train, random_state=1)
    prediction = model.predict(X_test)

    print_prediction_details(prediction)
    accuracy = print_model_results(y_test, prediction, random_state=1)
    compare_all_ones_baseline(y_test, accuracy, random_state=1)
    compare_random_state_zero(data, kept_predictors)


if __name__ == "__main__":
    main()
