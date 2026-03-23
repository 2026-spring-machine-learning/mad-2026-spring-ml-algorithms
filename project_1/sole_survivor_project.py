import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import seaborn as sns


#Function that finds which variables from the predictors are correlated
def predictor_correlation(predictors_df: pd.DataFrame, threshold: float = 0.8) -> None:

    predictor_correlation = predictors_df.corr()
    sns.heatmap(predictor_correlation, annot=True, mask=(predictor_correlation <= threshold))
    plt.show()

#Function that drops the variables from the predictors that are correlated
def drop_correlated_variables(predictors):
    number_predictors = predictors.drop(columns=['Teamwork'])
    number_predictors = number_predictors.drop(columns=['Adaptability'])
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
As for if the survivalist did a good job or not, I am a little bit conflicted to give a definitive answer. Although the R² value shows that all of the variables, when analyzed together, provide a reasonable prediction of the final SurvivalScore, the heatmap shows that each specific variable does not have a strong correlation to Survival Score. This could mean one of two things. One: The specific variables do not have a strong correlation, but when analyzing the variables all together will give you a stronger idea on what the final Survival Score will be. Two: There is no strong correlation and the observed relationships could be influenced by chance. I would almost recommend analyzing data from another season, if possible, to get a better understanding if the survivalists are doing a good job or not. 

"""