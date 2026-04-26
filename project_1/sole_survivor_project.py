import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import seaborn as sns

#r is different than r2. The error is with r2.r
#HEATMAP: needs to be above .3 (REMOVE LEADERSHIP)
#change briefing.
#sole-survivor-2


#Function that finds which variables from the predictors are correlated
def predictor_correlation(predictors_df: pd.DataFrame, threshold: float = 0.8) -> None:

    predictor_correlation = predictors_df.corr()
    sns.heatmap(predictor_correlation, annot=True, mask=(predictor_correlation <= threshold))
    plt.show()

#Function that drops the variables from the predictors that are correlated OR have an r value in the heatmap between -.3 and .3
def drop_correlated_variables(predictors):
    number_predictors = predictors.drop(columns=['Teamwork'])
    number_predictors = number_predictors.drop(columns=['Adaptability'])
    number_predictors = number_predictors.drop(columns= 'Leadership')
    return number_predictors

def analyze_past(sole_survivor_past_df: pd.DataFrame) -> None:

    number_predictors = sole_survivor_past_df.drop(columns=['Name'])
    number_predictors = drop_correlated_variables(number_predictors)

    predictors = number_predictors.drop(columns='SurvivalScore')


    predictor_correlation(predictors, threshold=.8)
    response_series = sole_survivor_past_df['SurvivalScore']

    #train and predict
    algorithm = lm.LinearRegression()
    model = algorithm.fit(predictors, response_series)

    #Heatmap to see correlation between predictors and Survival Score
    plt.figure(figsize=(8, 6))
    sns.heatmap(number_predictors.corr()[['SurvivalScore']], annot=True)
    plt.show()

  
    #Linear Regression model used for future seasons
    model = lm.LinearRegression().fit(predictors, response_series)

    #Calculate r-squared. 
    r_squared = model.score(predictors, response_series)
    print(f"r^2: {r_squared}")

    return model


#Part 2: Linear Regression:

def analyze_future(model, sole_survivor_future_df: pd.DataFrame):

    next_predictors = sole_survivor_future_df.drop(columns=['Name'])
    next_predictors = drop_correlated_variables(next_predictors)
    
    #Use predicted linear regression from previous season
    predictions = model.predict(next_predictors)
    results = sole_survivor_future_df.copy()
    results['PredictedSurvivalScore'] = predictions.round(2)

    #Predict the next top 3 contestants
    top_3 = results.sort_values(by='PredictedSurvivalScore', ascending=False).head(3)

    print("Top 3 Predicted Survivors:")
    print(top_3[['Name', 'PredictedSurvivalScore']])
    

def main():

    sole_survivor_past = pd.read_csv(r"mad-2026-spring-ml-algorithms\project_1\sole_survivor_past.csv")
    sole_survivor_next = pd.read_csv(r"mad-2026-spring-ml-algorithms\project_1\sole_survivor_next.csv")
    model = analyze_past(sole_survivor_past)
    analyze_future(model, sole_survivor_next)

if __name__ == "__main__":
    main()


""" Briefing :
The survivalist did do a good job predicting. The R² value (which is .8078...) shows that all of the variables, when analyzed together, provide a reasonable prediction of the final Survival Score.
"""