import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.metrics as metrics
import sklearn.model_selection as ms
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Removed resourcefulness as it showed nearly the same correlation as adaptability,
# Adaptability had a slightly stronger correlation with SurvivalScore, so it was kept.
def select_features(survive_df: pd.DataFrame):
    
    predictors_df = survive_df[['Leadership', 'MentalToughness', 'SurvivalSkills', 'RiskTaking', 'Adaptability', 'PhysicalFitness', 'Teamwork', 'Stubbornness']]

    response_series = survive_df['SurvivalScore'].to_numpy()

# Training and predict
    algorithm = lm.LinearRegression()
    model = algorithm.fit(predictors_df, response_series)

# r squared:
    r_squared = model.score(predictors_df, response_series)
    print(f"R^2: {r_squared}")

    print(response_series)
    return (predictors_df, response_series)

def response_correlation(predictors_df, pastsurvivor_df):

    predictors_response_df = pd.concat([predictors_df, pastsurvivor_df['SurvivalScore']], axis='columns')
    print(predictors_response_df)

    correlation_matrix = predictors_response_df.corr()
    print(f"Correlation Matrix {correlation_matrix}")

    volume_correlation_matrix = correlation_matrix[['SurvivalScore']].sort_values(by='SurvivalScore', ascending=False)

    sns.heatmap(volume_correlation_matrix, annot=True)
    plt.show()

    return predictors_df

def analyze(predictor_df):

    corr_vals = predictor_df.corr()
    print(f"Check:{corr_vals}")

    mask_low_corr = corr_vals <= 0.8
    sns.heatmap(corr_vals, mask=mask_low_corr, annot=True)
    plt.show()

    return predictor_df

## Training the model on past data only and returns trained movel
def predict(survivor_df: pd.DataFrame):

    predictors_df, response_series = select_features(survivor_df)

    (X_train, X_test, y_train, y_test) = ms.train_test_split(
            predictors_df, response_series, test_size=0.2
        )
    
    
    algorithm = lm.LinearRegression()
    model = algorithm.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Prediction and Result:")
    comparison_df = pd.DataFrame({"Predicted Score": predictions,"Actual Score": y_test})
    print(comparison_df)
    #showing rmse:
    mse_value = metrics.mean_squared_error(y_test, predictions)
    rmse_value = np.sqrt(mse_value)
    print(f"RMSE: {rmse_value}")

    test_r2 = model.score(X_test, y_test)
    print(f"Test r^2: {test_r2}") 

    return model, predictions, X_test

## This Will predict on next seasons contestants
def predict_next_season(model, next_season_df):

    predictors_df = next_season_df[['Leadership', 'MentalToughness', 'SurvivalSkills', 'RiskTaking', 'Adaptability', 'PhysicalFitness', 'Teamwork', 'Stubbornness']]

    predictions = model.predict(predictors_df)

    next_season_df = next_season_df.copy()
    next_season_df['PredictedSurvivalScore'] = predictions

    print("Next Season Predictions:")
    print(next_season_df[['Name', 'PredictedSurvivalScore']])

    return next_season_df

def top_3_survivors(next_season_df):

    top_3 = next_season_df.sort_values(by='PredictedSurvivalScore', ascending=False).head(3)

    print("Top 3 Predicted Survivors For Next Season:")
    print(top_3[['Name', 'PredictedSurvivalScore']])

    return top_3[['Name', 'PredictedSurvivalScore']]

def main():
#Read in the CSV
    pastsurvivor_df = pd.read_csv('sole_survivor_past.csv')
    nextszn_survivor = pd.read_csv('sole_survivor_next.csv')
# Used for training
    predictors_df, response_series = select_features(pastsurvivor_df)
    response_correlation(predictors_df, pastsurvivor_df)
    analyze(predictors_df)

    model, predictions, X_test = predict(pastsurvivor_df)
# Final predictions for next season
    next_season_predictions = predict_next_season(model, nextszn_survivor)
#Top 3 projected for next season:
    top_3_survivors(next_season_predictions)
if __name__ == "__main__":
    main()