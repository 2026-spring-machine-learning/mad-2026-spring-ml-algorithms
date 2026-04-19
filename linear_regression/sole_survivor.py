import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.metrics as metrics

# Add plotting imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns



##This function tests correlation between the indepedent variables. It is called by feature_select
def testing_ind(predictors_df):
    correlation_matrix = predictors_df.corr()
    print(f'This is the correlation matrix \n {correlation_matrix}')
    cor_mask = correlation_matrix <= 0.8
    
    sns.heatmap(correlation_matrix, mask=cor_mask, annot=True)
    plt.show()

    return predictors_df


##This function is called by predict and is used to determine which independent variables to use in the model. It also checks for linearity and independence of the predictors.
    # Independent Variables avail = 'Leadership', 'MentalToughness', 'SurvivalSkills', 'RiskTaking', 'Resourcefulness', 'Adaptability', 'PhysicalFitness', 'Teamwork', 'Stubbornness'
    # Dependant Variable 'SurvivalScore'
def feature_select(survivor_df: pd.DataFrame) -> None:

    predictors_df = survivor_df[['MentalToughness', 'SurvivalSkills', 'Adaptability', 'PhysicalFitness']]
    response_series = survivor_df['SurvivalScore'].to_numpy()
    

    predictors_and_response_df = pd.concat([predictors_df, survivor_df['SurvivalScore']], axis = 'columns')
    print(predictors_and_response_df)
    correlation_matrix = predictors_and_response_df.corr()
    print(f"This is the correlation matrix \n {correlation_matrix}")
    volume_correlation_matrix = correlation_matrix[['SurvivalScore']].sort_values(by='SurvivalScore', ascending=False)

    sns.heatmap(volume_correlation_matrix, annot=True)
    plt.show()

    # Test independence between predictors
    testing_ind(predictors_df)

    # note: not using the 'best-fit' model because we have multiple Independent Variables. 
    return (predictors_df, response_series)


##This fuction is called by main and is used to train the model, make predictions, and evaluate the model. It also returns the model, predictors_testing_df, and prediction for use in the top_three function.
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

##This function is called by main and is used to determine the top three predicted survivors based on the model's predictions.
##Note: consulted Claude Haiku for help on this function. I understand the logic but was having trouble with the implementation.
def top_three(model, next_survivor_df):
    ##Makes predictions on the full next_survivor_df and returns top 3
    # Extract predictors from next_survivor_df
    next_predictors_df = next_survivor_df[['MentalToughness', 'SurvivalSkills', 'Adaptability', 'PhysicalFitness']]
    
    # Make predictions on full dataset
    predictions = model.predict(next_predictors_df)
    
    # Add predictions to dataframe
    next_survivor_df_copy = next_survivor_df.copy()
    next_survivor_df_copy['PredictedScore'] = predictions
    
    # Sort by predicted score and get top 3
    top3 = next_survivor_df_copy.sort_values(by='PredictedScore', ascending=False).head(3)

    print("\nTop 3 Predicted Survivors:")
    print(top3[['Name', 'PredictedScore']])

    return top3[['Name', 'PredictedScore']]



def main():

    ##Loading the data
        ##Note: path may need to change to run. Can only run on my local machine this way. 
    survivor_df = pd.read_csv("H:/MATC/3_2026 Spring Sem/MachineLearning/mad-2026-spring-ml-algorithms/linear_regression/sole_survivor_past.csv")
    next_survivor_df = pd.read_csv("H:/MATC/3_2026 Spring Sem/MachineLearning/mad-2026-spring-ml-algorithms/linear_regression/sole_survivor_next.csv")

    ##Model training, predicting and evaluation
    model, predictors_testing_df, prediction = predict(survivor_df)

    top_three(model, next_survivor_df)

if __name__ == "__main__":
    main()
