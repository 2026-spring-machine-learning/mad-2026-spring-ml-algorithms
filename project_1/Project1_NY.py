import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import seaborn as sns


def correlation_heatmap(df: pd.DataFrame, cutoff: float = 0.8) -> None:
    corr_matrix = df.corr()
    mask = corr_matrix <= cutoff
    sns.heatmap(corr_matrix, annot=True, mask=mask)
    plt.show()

def filter_features(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.drop(columns=['Teamwork'])
    filtered = filtered.drop(columns=['Adaptability'])
    return filtered

def training_model(past_df: pd.DataFrame):
    training_df = past_df.drop(columns=['Name'])
    training_df = filter_features(training_df)

    X = training_df.drop(columns=['SurvivalScore'])
    y = training_df['SurvivalScore']

    correlation_heatmap(X, cutoff=0.8)

    model = lm.LinearRegression()
    model.fit(X, y)

    plt.figure(figsize=(8, 6))
    sns.heatmap(training_df.corr()[['SurvivalScore']], annot=True)
    plt.show()

    r2 = model.score(X, y)
    print(f"r^2: {r2}")
    return model

def predict_survivors(model, future_df: pd.DataFrame):

    future_data = future_df.drop(columns=['Name'])
    future_data = filter_features(future_data)
    preds = model.predict(future_data)
    output = future_df.copy()
    output['PredictedSurvivalScore'] = preds.round(2)

    top_candidates = output.sort_values(
        by='PredictedSurvivalScore',
        ascending=False
    ).head(3)

    print("Top 3 Predicted Survivors:")
    print(top_candidates[['Name', 'PredictedSurvivalScore']])

def main():
    past_path = r"C:\Users\nicky\OneDrive - Madison College\Machine Learning\mad-2026-spring-ml-algorithms\project_1\sole_survivor_past.csv"
    future_path = r"C:\Users\nicky\OneDrive - Madison College\Machine Learning\mad-2026-spring-ml-algorithms\project_1\sole_survivor_next.csv"

    past_df = pd.read_csv(past_path)
    future_df = pd.read_csv(future_path)

    model = training_model(past_df)
    predict_survivors(model, future_df)


if __name__ == "__main__":
    main()