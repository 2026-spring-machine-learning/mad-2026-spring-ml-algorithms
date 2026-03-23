import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.metrics as metrics

# Add plotting imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns


##This function checks the results of the sole_survivor_past.csv in order to determine if the indvidual stats are comparable to the SurvivalScore.
    # Independent Variables = 'Leadership', 'MentalToughness', 'SurvivalSkills', 'RiskTaking', 'Resourcefulness', 'Adaptability', 'PhysicalFitness', 'Teamwork', 'Stubbornness'
    # Dependant Variable 'SurvivalScore'

def feature_select(survivor_df: pd.DataFrame) -> None:

    predictors_df = survivor_df[['MentalToughness', 'SurvivalSkills', 'Adaptability', 'PhysicalFitness', 'Teamwork', 'Stubbornness']]
    response_series = survivor_df['SurvivalScore'].to_numpy()
    
    # Train and predict. Needed to check R^2.
    algorithm = lm.LinearRegression()
    model = algorithm.fit(predictors_df, response_series)

    # Calculate r-squared.
    ## R^2 is 0.82 which is above the 0.3 threashold. This tells us that the 'SurvivalScore' is in line with the categories rated. 
    r_squared = model.score(predictors_df, response_series)
    print(f"r^2: {r_squared}")

    # note: not using the 'best-fit' model because we have multiple Independent Variables. 
    print(response_series)
    return (predictors_df, response_series)


## This function tests linearity by looking at Independent Variabls vs the Dependent Variable. 
def testing_linearity(predictors_df, survivor_df):


    predictors_and_response_df = pd.concat([predictors_df, survivor_df['SurvivalScore']], axis = 'columns')
    print(predictors_and_response_df)
    correlation_matrix = predictors_and_response_df.corr()
    print(f"This is the correlation matrix \n {correlation_matrix}")
    volume_correlation_matrix = correlation_matrix[['SurvivalScore']].sort_values(by='SurvivalScore', ascending=False)

    sns.heatmap(volume_correlation_matrix, annot=True)
    plt.show()

    return predictors_df


##This function tests correlation between the indepedent variables. 
def testing_ind(predictors_df):
    correlation_matrix = predictors_df.corr()
    print(f'This is the correlation matrix \n {correlation_matrix}')
    cor_mask = correlation_matrix <= 0.8
    
    sns.heatmap(correlation_matrix, mask=cor_mask, annot=True)
    plt.show()

    return predictors_df


def predict(survivor_df: pd.DataFrame):

    predictors_df, response_series = feature_select(survivor_df)

    
    (predictors_training_df, predictors_testing_df, \
        response_training_df, response_testing_df) \
            = ms.train_test_split(predictors_df, response_series, test_size=0.2)

    ##train and predict
    algorithm = lm.LinearRegression()
    model = algorithm.fit(predictors_training_df, response_training_df)

    # 4. Predict on the test set
    prediction = model.predict(predictors_testing_df)

    print("Prediction vs Actual:")
    print(pd.DataFrame({
        "Predicted": prediction,
        "Actual": response_testing_df
    }))

    ##Checking RMSE and R2 on Testing sets
    mse = metrics.mean_squared_error(response_testing_df, prediction)
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse}")

    rsquare_test = model.score(predictors_testing_df, response_testing_df)
    print(f"Test r^2: {rsquare_test}")

    return model, predictors_testing_df, prediction

def top_three(survivor_df, predictors_testing_df, prediction):
    ##Note: consulted CoPilot for help how to do this
    
    # Reset index to ensure data goes to right person.
    survivor_df_reset = survivor_df.reset_index(drop=True)
    predictors_testing_df_reset = predictors_testing_df.reset_index(drop=True)
    matching_rows = survivor_df_reset.loc[predictors_testing_df_reset.index].copy()

    # Get the predicted scores
    matching_rows['PredictedScore'] = prediction

    # Sort by predicted score
    top3 = matching_rows.sort_values(by='PredictedScore', ascending=False).head(3)

    print("\nTop 3 Predicted Survivors:")
    print(top3[['Name', 'PredictedScore']])

    return top3[['Name', 'PredictedScore']]



def main():

    ##Loading the data
        ##Note: path may need to change to run. Can only run on my local machine this way. 
    survivor_df = pd.read_csv("H:/MATC/3_2026 Spring Sem/MachineLearning/mad-2026-spring-ml-algorithms/linear_regression/sole_survivor_past.csv")
    next_survivor_df = pd.read_csv("H:/MATC/3_2026 Spring Sem/MachineLearning/mad-2026-spring-ml-algorithms/linear_regression/sole_survivor_next.csv")
    
    ##feature Selection returns predictors and response array
    predictors_df, response_series = feature_select(survivor_df) 

    ##Tests Linearity and correlations
    testing = testing_linearity(predictors_df, survivor_df)
    test_individuality = testing_ind(predictors_df)

    ##Model training, predicting and evaluation
    model, predictors_testing_df, prediction = predict(survivor_df)

    top_three(survivor_df, predictors_testing_df, prediction)

if __name__ == "__main__":
    main()
