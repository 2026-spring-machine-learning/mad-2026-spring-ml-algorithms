import os

import sklearn.model_selection as ms

from sole_survivor.data_loader import FEATURES, get_features_and_target, load_next_data, load_past_data
from sole_survivor.model import evaluate_model, get_statistical_summary, train_model
from sole_survivor.predictor import get_top_n, predict_next_season
from sole_survivor.visualizer import (
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_feature_vs_target,
    plot_predicted_vs_actual,
    plot_residuals,
)

DATA_DIR = os.path.dirname(__file__)
PAST_CSV = os.path.join(DATA_DIR, "sole_survivor_past.csv")
NEXT_CSV = os.path.join(DATA_DIR, "sole_survivor_next.csv")


def main():
    # ── 1. Data ingestion ───────────────────────────────────────────
    past_df = load_past_data(PAST_CSV)
    next_df = load_next_data(NEXT_CSV)

    print(f"Past data: {past_df.shape[0]} rows, {past_df.shape[1]} columns")
    print(f"Next data: {next_df.shape[0]} rows, {next_df.shape[1]} columns\n")

    # ── 2. Exploratory data analysis ────────────────────────────────
    plot_correlation_heatmap(past_df)
    plot_feature_vs_target(past_df)

    # ── 3. Train / test split ───────────────────────────────────────
    X, y = get_features_and_target(past_df)
    X_train, X_test, y_train, y_test = ms.train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 4. Model training & evaluation ──────────────────────────────
    model = train_model(X_train, y_train)
    eval_metrics = evaluate_model(model, X_test, y_test)

    # ── 5. Statistical validity (statsmodels OLS) ───────────────────
    ols_model = get_statistical_summary(X, y, FEATURES)

    # ── 6. Validation plots ─────────────────────────────────────────
    y_pred = model.predict(X_test)
    plot_predicted_vs_actual(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_feature_importance(model, FEATURES)

    # ── 7. Predict next season & rank ───────────────────────────────
    predicted_df = predict_next_season(model, next_df)
    top_3 = get_top_n(predicted_df, n=3)

    print("\n=== All Next-Season Predictions ===")
    print(
        predicted_df[["Name", "PredictedSurvivalScore"]]
        .sort_values("PredictedSurvivalScore", ascending=False)
        .to_string(index=False)
    )

    print("\n=== Top 3 Predicted Winners ===")
    for rank, row in enumerate(top_3.itertuples(), start=1):
        print(f"  #{rank}  {row.Name:15s}  predicted score: {row.PredictedSurvivalScore:.2f}")


if __name__ == "__main__":
    main()
