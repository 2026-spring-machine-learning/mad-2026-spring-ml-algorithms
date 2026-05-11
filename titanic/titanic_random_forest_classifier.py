from pathlib import Path

import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "Titanic-Dataset.csv"
REPORT_FILE = SCRIPT_DIR / "titanic_cross_validation_accuracy.txt"
NUMERIC_PREDICTORS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
RESPONSE = "Survived"
MODEL_NAME = "Random Forest"
RANDOM_STATE = 1
N_SPLITS = 10


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
    """Binary-encode Sex and one-hot encode Embarked."""
    df = df.copy()
    df["Sex_encoded"] = df["Sex"].map({"male": 0, "female": 1})

    embarked_dummies = (
        pd.get_dummies(df["Embarked"], prefix="Embarked", drop_first=True).astype(int)
    )
    df = pd.concat([df, embarked_dummies], axis=1)

    cat_cols = ["Sex_encoded"] + list(embarked_dummies.columns)
    print(f"\nEncoded columns added: {cat_cols}")
    print(
        "  Sex_encoded  -> male=0, female=1  | counts: "
        f"{df['Sex_encoded'].value_counts().to_dict()}"
    )
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


def evaluate_with_cross_validation(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    print("\n" + "=" * 50)
    print(f"{N_SPLITS}-FOLD CROSS-VALIDATION")
    print("=" * 50)
    for fold_number, score in enumerate(scores, start=1):
        print(f"Fold {fold_number:>2} accuracy: {score:.4f}")

    mean_accuracy = scores.mean()
    print(f"\nMean cross-validation accuracy: {mean_accuracy:.4f}")
    print(f"Accuracy standard deviation:    {scores.std():.4f}")
    return scores, mean_accuracy


def write_accuracy_report(scores, mean_accuracy, predictors):
    report_lines = [
        "Titanic Random Forest 10-Fold Cross-Validation",
        f"Predictors used: {', '.join(predictors)}",
        f"Fold accuracies: {', '.join(f'{score:.4f}' for score in scores)}",
        f"Mean accuracy: {mean_accuracy:.4f}",
    ]
    REPORT_FILE.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nAccuracy report written to: {REPORT_FILE.name}")


def main():
    df = load_and_clean_data()
    df, cat_cols = encode_categorical(df)

    all_predictors = NUMERIC_PREDICTORS + cat_cols
    data = df[all_predictors + [RESPONSE]].copy()
    kept_predictors = select_features(data, all_predictors)

    X = data[kept_predictors]
    y = data[RESPONSE]
    print(f"\nRows used for cross-validation: {len(X)}")
    print(f"Predictor count: {X.shape[1]}")

    scores, mean_accuracy = evaluate_with_cross_validation(X, y)
    write_accuracy_report(scores, mean_accuracy, kept_predictors)


if __name__ == "__main__":
    main()
