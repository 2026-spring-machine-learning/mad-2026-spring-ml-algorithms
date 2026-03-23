import numpy as np
import pandas as pd
import sklearn.linear_model as lm

# Add plotting imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns


##step 1 - checking the results of the sole_survivor_past.csv 
    #  Independent Variables = 'Leadership', 'MentalToughness', 'SurvivalSkills', 'RiskTaking', 'Resourcefulness', 'Adaptability', 'PhysicalFitness', 'Teamwork', 'Stubbornness'

    # Dependant Variable 'SurvivalScore'

def predict(survivor_df: pd.DataFrame) -> None:

    predictors_df = survivor_df[['MentalToughness', 'SurvivalSkills', 'Adaptability', 'PhysicalFitness', 'Teamwork', 'Stubbornness']]

    response_series = survivor_df['SurvivalScore'].to_numpy()
    
    # Train and predict.
    algorithm = lm.LinearRegression()
    model = algorithm.fit(predictors_df, response_series)
    prediction = model.predict(predictors_df)

    # Calculate r-squared.
    ## R^2 is 0.82 which is above the 0.3 threashold. This tells us that the 'SurvivalScore' is in line with the categories rated. 
    r_squared = model.score(predictors_df, response_series)
    print(f"r^2: {r_squared}")

    # note: not using the 'best-fit' model because we have multiple Independent Variables. 


    return (predictors_df, survivor_df)


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


def main():

    ##Note: path may need to change to run. Can only run on my local machine this way. 
    survivor_df = pd.read_csv("H:/MATC/3_2026 Spring Sem/MachineLearning/mad-2026-spring-ml-algorithms/linear_regression/sole_survivor_past.csv")
    (predictors_df, survivor_df) = predict(survivor_df) 

    testing = testing_linearity(predictors_df, survivor_df)
    test_individuality = testing_ind(predictors_df)




if __name__ == "__main__":
    main()
