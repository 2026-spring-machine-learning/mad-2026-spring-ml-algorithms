import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
from sklearn.metrics import accuracy_score

def read_and_clean():
    titanic_df = pd.read_csv('Titanic/Titanic-Dataset.csv') 
    titanic_df = titanic_df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()
    print(titanic_df.head())
    return titanic_df

def feature_select(df):
    predictors = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
    response_series = df['Survived']

    correlation_matrix = df.corr(numeric_only=True)
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    response_correlation = correlation_matrix['Survived'].drop('Survived')
    print("correlated with survival:")
    print(response_correlation)

    selected_predictors = response_correlation[response_correlation.abs() >= 0.1].index.tolist()
    print("\nSelected Predictors:")
    print(selected_predictors)

    predictors = df[selected_predictors]
    return predictors, response_series

def perform_logistic_regression(predictors_df, response_series):
    predictors_training, predictors_testing, response_training, response_testing = ms.train_test_split(
        predictors_df,
        response_series,
        test_size=0.2,
        random_state=0   # using 0 to compare against the all-1s baseline
    )

    algorithm = lm.LogisticRegression(max_iter=1000)
    model = algorithm.fit(predictors_training, response_training)
    predictions = model.predict(predictors_testing)

    print("\nPredictions:")
    print(predictions)

    print("\nReal Values:")
    print(response_testing.to_numpy())

    print("\nLogistic Regression Accuracy:")
    print(accuracy_score(response_testing, predictions))

    all_ones = np.ones(len(response_testing), dtype=int)
    print("\nAll 1s baseline accuracy:")
    print(accuracy_score(response_testing, all_ones))

def main():
    df = read_and_clean()
    predictors_df, response_series = feature_select(df)
    perform_logistic_regression(predictors_df, response_series)


if __name__ == "__main__":
    main()