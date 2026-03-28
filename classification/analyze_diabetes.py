import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm


def read_diabetes():
     """Read the pa_diabetes.csv file into a pandas DataFrame."""
     df = pd.read_csv("pa_diabetes.csv")
     return df


def analyze(diabetes_df):
    # diabetes_df = ['3']
    diabetes_df = diabetes_df.drop(['DNR Order', 'Med Tech'], axis='columns')
    predictors = diabetes_df.drop('Outcome', axis='columns')
    response = diabetes_df['Outcome']
    # print(predictors.columns)
    (diabetes_predictors_training, diabetes_predictors_testing,
     diabetes_response_training, diabetes_response_testing) = \
        ms.train_test_split(predictors, response, test_size=0.2,
        random_state=4) #, random_state=0)
    # print("Training Predictors:")
    # print(diabetes_predictors_training)
    # print("Training Response:")
    # print(diabetes_response_training)
    # print("Testing Predictors:")
    # print(diabetes_predictors_testing)
    # print("Testing Response:")
    # print(diabetes_response_testing)

    algorithm = lm.LogisticRegression()
    model = algorithm.fit(diabetes_predictors_training, diabetes_response_training)
    predictions = model.predict(diabetes_predictors_testing)
    print("Predictions:")
    print(predictions)
    print("Actual:")
    print(diabetes_response_testing.values)
    print("Comparison:")
    print(predictions == diabetes_response_testing.values)
    print("Accuracy:")
    print((predictions == diabetes_response_testing.values).mean())


def main():
    df = read_diabetes()
    print(df.head())
    analyze(df)


if __name__ == "__main__":
    main()
