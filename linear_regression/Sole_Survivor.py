import numpy as np 
import pandas as pd 
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import sklearn.metrics as metrics
import sklearn.model_selection as ms

def feature_select(survivor_df):
    predictors_df = survivor_df[[
        'Leadership',
        'MentalToughness',
        'SurvivalSkills',
        'RiskTaking',
        'PhysicalFitness',
        'Teamwork',
        'Stubbornness'
    ]]

    response_series = survivor_df['SurvivalScore'].to_numpy()

    #Correlation
    predictors_and_response_df = pd.concat([predictors_df, survivor_df[['SurvivalScore']]], axis='columns')

    correlation_matrix = predictors_and_response_df.corr()
    survival_correlation = correlation_matrix[['SurvivalScore']].sort_values(by='SurvivalScore', ascending=False)
    print(survival_correlation)
    print(correlation_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix[['SurvivalScore']], annot=True)
    plt.title("Correlation with SurvivalScore")
    plt.show()

    return (predictors_df, response_series)

def predict(df):
    print(df)

    (predictors_df, response_series) = feature_select(df)

    (predictors_training_df, predictors_testing_df,
     score_training, score_testing) = ms.train_test_split(
         predictors_df, response_series, test_size=0.2
     )
    
    algorithm = lm.LinearRegression()
    model = algorithm.fit(predictors_training_df, score_training)
    prediction = model.predict(predictors_testing_df)

    print("\n Prediction vs Actual")
    print(prediction)
    print(score_testing)

    mse = metrics.mean_squared_error(score_testing, prediction)
    rmse = np.sqrt(mse)
    print(f"\nRMSE: {rmse}")

    r_squared = model.score(predictors_df, response_series)
    print(f"r^2: {r_squared}")

    return model

def predict_next_season(model, next_season):
    predictors_df = next_season[[
        "Leadership",
        "MentalToughness",
        "SurvivalSkills",
        "RiskTaking",
        "PhysicalFitness",
        "Teamwork",
        "Stubbornness"
    ]]

    predictions = model.predict(predictors_df)
    next_season["PredictedSurvivalScore"] = predictions

    print("Next Season Predictions")
    print(next_season)

    top_3 = next_season.sort_values(by="PredictedSurvivalScore", ascending=False).head(3)

    print("Top 3 Likely Winners:")
    print(top_3[['Name', 'PredictedSurvivalScore']])

    return (next_season, top_3)

def main():
    past_survivor = pd.read_csv('sole_survivor_past.csv')
    new_survivor = pd.read_csv('sole_survivor_next.csv')
    #print(past_survivor)
    #print(new_survivor)
    #print(past_survivor.describe())
    
    model = predict(past_survivor)

    predict_next_season(model, new_survivor)

if __name__ == "__main__":
    main()

"I would say there's definetly some unpredictability on this. I think the correlations and r^2 are reasonable, and overall, the findings seem valid but certainly not perfect."