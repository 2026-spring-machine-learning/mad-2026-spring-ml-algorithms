import pandas as pd

from sole_survivor.data_loader import FEATURES


def predict_next_season(model, next_df):
    X_next = next_df[FEATURES]
    predictions = model.predict(X_next)

    result_df = next_df.copy()
    result_df["PredictedSurvivalScore"] = predictions
    return result_df


def get_top_n(predicted_df, n=3):
    top = (
        predicted_df[["Name", "PredictedSurvivalScore"]]
        .sort_values("PredictedSurvivalScore", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    return top
