import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import imblearn.over_sampling as ios


def read_diabetes():
     """Read the pa_diabetes.csv file into a pandas DataFrame."""
     df = pd.read_csv("pa_diabetes.csv")
     return df


def show_accuracy(diabetes_response_testing, predictions):
    print(f"Accuracy: {(predictions == diabetes_response_testing.values).mean()}")
    # print((predictions == diabetes_response_testing.values).mean())


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
        # print("Training Predictors:")
        # print(diabetes_predictors_training)
        # print("Training Response:")
        # print(diabetes_response_training)
        # print("Testing Predictors:")
        # print(diabetes_predictors_testing)
        # print("Testing Response:")
        # print(diabetes_response_testing)

        algorithm = lm.LogisticRegression(max_iter=100000)
        model = algorithm.fit(diabetes_predictors_training, diabetes_response_training)
        predictions = model.predict(diabetes_predictors_testing)
        # print("Predictions:")
        # print(predictions)
        # print("Actual:")
        # print(diabetes_response_testing.values)
        # print("Comparison:")
        # print(predictions == diabetes_response_testing.values)
        # all_zeros = np.zeros_like(diabetes_response_testing)
        # print(f"all_zeros: {all_zeros}")
        show_accuracy(diabetes_response_testing, predictions)
        # show_accuracy(diabetes_response_testing, all_zeros)


def analyze(diabetes_df):
    # diabetes_df = ['3']
    diabetes_df = diabetes_df.drop(['DNR Order', 'Med Tech'], axis='columns')
    predictors = diabetes_df.drop('Outcome', axis='columns')
    response = diabetes_df['Outcome']
    # print(predictors.columns)

    for balance_counter in range(2):
        if balance_counter == 0:
            print("Unbalanced:")
        else:
            print("Balanced:")
        perform_logistic_regression(predictors, response, balance_counter)


def main():
    df = read_diabetes()
    # print(df.head())
    analyze(df)


if __name__ == "__main__":
    main()
