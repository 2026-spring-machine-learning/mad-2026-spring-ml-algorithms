import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import imblearn.over_sampling as ios

##04/15/26 -- Not done yet!

def read_and_clean_df():
     df = pd.read_csv("H:/MATC/3_2026 Spring Sem/MachineLearning/mad-2026-spring-ml-algorithms/classification/Titanic-Dataset.csv")
     df = df.dropna()
     print(df.head())
     return df


def show_accuracy(df_response_testing, predictions):
    print(f"Accuracy: {(predictions == df_response_testing.values).mean()}")
    # print((predictions == df_response_testing.values).mean())


def perform_logistic_regression(predictors, response, balance_counter):
    for random_state in range(0, 3):
        if balance_counter == 1:
            random_over_sampler = ios.RandomOverSampler(random_state=random_state)
            predictors, response \
                = random_over_sampler.fit_resample(predictors, response)

        (diabetes_predictors_training, diabetes_predictors_testing,
        diabetes_response_training, diabetes_response_testing) = \
            ms.train_test_split(predictors, response, test_size=0.2,
            random_state=random_state) #, random_state=0)


        algorithm = lm.LogisticRegression(max_iter=100000)
        model = algorithm.fit(diabetes_predictors_training, diabetes_response_training)
        predictions = model.predict(diabetes_predictors_testing)

        show_accuracy(diabetes_response_testing, predictions)



def analyze(df):
    predictors = df[['Pclass', 'Age', 'SibSp', 'Fare']]
    response = df['Survived']
    # print(predictors.columns)

    for balance_counter in range(2):
        if balance_counter == 0:
            print("Unbalanced:")
        else:
            print("Balanced:")
        perform_logistic_regression(predictors, response, balance_counter)


def main():
    df = read_and_clean_df()
    # print(df.head())
    analyze(df)


if __name__ == "__main__":
    main()
