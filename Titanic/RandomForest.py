import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def read_and_clean():
    titanic_df = pd.read_csv('Titanic/Titanic-Dataset.csv')
    titanic_df = titanic_df[
        ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    ].dropna()

    print(titanic_df.head())
    return titanic_df

def encode_columns(df):
    encoded_df = df.copy()

    
    encoded_df['Sex'] = encoded_df['Sex'].map({'male': 0, 'female': 1})

    
    embarked_dummies = pd.get_dummies(encoded_df['Embarked'], prefix='Embarked')
    encoded_df = pd.concat([encoded_df, embarked_dummies], axis=1)
    encoded_df = encoded_df.drop(columns=['Embarked'])

    print("\nEncoded Data:")
    print(encoded_df.head())

    return encoded_df

def feature_select(df):
    predictors = df.drop(columns=['Survived'])
    response_series = df['Survived']

    correlation_matrix = df.corr(numeric_only=True)
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    response_correlation = correlation_matrix['Survived'].drop('Survived')
    print("\nCorrelated with Survival:")
    print(response_correlation)

    selected_predictors = response_correlation[response_correlation.abs() >= 0.1].index.tolist()
    print("\nSelected Predictors:")
    print(selected_predictors)

    predictors = df[selected_predictors]
    return predictors, response_series

def perform_cross_validate(predictors_df, response_series):
    algorithm = RandomForestClassifier(n_estimators=100, random_state=1)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(algorithm, predictors_df, response_series, cv=kfold, scoring='accuracy')

    print("\nRandom Forest 10-fold Cross-validation scores:")
    print(scores)
    print("\nMean Accuracy:")
    print(scores.mean())

# def show_results(response_testing, predictions, model_name):
#     print(f"\n{model_name} Predicted Values:")
#     print(predictions)

#     print("\nActual Values:")
#     print(response_testing.to_numpy())

#     print(f"\n{model_name} Accuracy:")
#     print(accuracy_score(response_testing, predictions))

#     all_ones = np.ones(len(response_testing), dtype=int)
#     print("\nAll 1s Accuracy:")
#     print(accuracy_score(response_testing, all_ones))

#     matrix = confusion_matrix(response_testing, predictions)
#     print("\nConfusion Matrix:")
#     print(matrix)

#     true_negatives = matrix[0][0]
#     true_positives = matrix[1][1]

#     print("\nTrue Negatives:")
#     print(true_negatives)

#     print("\nTrue Positives:")
#     print(true_positives)

# def perform_logistic_regression(predictors_df, response_series):
#     predictors_training, predictors_testing, response_training, response_testing = ms.train_test_split(
#         predictors_df,
#         response_series,
#         test_size=0.2,
#         random_state=1
#     )

#     algorithm = lm.LogisticRegression(max_iter=100000)
#     model = algorithm.fit(predictors_training, response_training)
#     predictions = model.predict(predictors_testing)

#     show_results(response_testing, predictions, "Logistic Regression")

# def perform_random_forest(predictors_df, response_series):
#     predictors_training, predictors_testing, response_training, response_testing = ms.train_test_split(
#         predictors_df,
#         response_series,
#         test_size=0.2,
#         random_state=1
#     )

#     algorithm = RandomForestClassifier(n_estimators=100, random_state=1)
#     model = algorithm.fit(predictors_training, response_training)
#     predictions = model.predict(predictors_testing)

#     show_results(response_testing, predictions, "Random Forest")

def main():
    df = read_and_clean()
    encoded_df = encode_columns(df)
    predictors_df, response_series = feature_select(encoded_df)

    perform_cross_validate(predictors_df, response_series)

    # perform_logistic_regression(predictors_df, response_series)
    # perform_random_forest(predictors_df, response_series)

if __name__ == "__main__":
    main()