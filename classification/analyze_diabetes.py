import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.ensemble as es
import sklearn.metrics as metrics
import imblearn.over_sampling as ios
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def compute_confusion_matrix_numbers(actual_data_df, prediction):
    confusion_tuple = metrics.confusion_matrix(actual_data_df, prediction)
    command_line_display_as_accuracy_top_confusion_matrix = confusion_tuple.T
    command_line_display_as_accuracy_top_confusion_matrix = np.flip(command_line_display_as_accuracy_top_confusion_matrix, axis=0)
    command_line_display_as_accuracy_top_confusion_matrix = np.flip(command_line_display_as_accuracy_top_confusion_matrix, axis=1)
    true_negs = 0
    false_poss = 0
    false_negs = 0
    true_poss = 0
    sensitivity = 0
    specificity = 0
    if len(confusion_tuple.ravel()) == 4:
        (true_negs, false_poss, false_negs, true_poss) = confusion_tuple.ravel()
        if ((true_poss + false_negs) > 0) and ((true_negs + false_poss) > 0):
            sensitivity = true_poss / (true_poss + false_negs)
            specificity = true_negs / (true_negs + false_poss)
    return (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs, false_poss, false_negs, true_poss, sensitivity, specificity)


def create_confusion_matrix(actual_data_df, prediction):
    (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs, false_poss, false_negs, true_poss, sensitivity, specificity) \
        = compute_confusion_matrix_numbers(actual_data_df, prediction)
    # if (sensitivity > 0) or (specificity > 0):
    #     print(f'tp: {true_poss}, fn: {false_negs}, tn: {true_negs}, fp: {false_poss}, sensitivity: {sensitivity}, specificity: {specificity}.')
    # print(command_line_display_as_accuracy_top_confusion_matrix)
    sns.heatmap(confusion_tuple, annot=True)
    plt.show()


def read_diabetes():
     """Read the pa_diabetes.csv file into a pandas DataFrame."""
     csv_path = Path(__file__).resolve().parent / "pa_diabetes.csv"
     df = pd.read_csv(csv_path)
     return df


def show_accuracy(diabetes_response_testing, predictions):
    print(f"Accuracy: {(predictions == diabetes_response_testing.values).mean()}")
    # print((predictions == diabetes_response_testing.values).mean())


def perform_logistic_regression(predictors, response, balance_counter):
    # for random_state in range(0, 3):
    #     if balance_counter == 1:
    #         random_over_sampler = ios.RandomOverSampler(random_state=random_state)
    #         predictors, response \
    #             = random_over_sampler.fit_resample(predictors, response)

    (diabetes_predictors_training, diabetes_predictors_testing,
    diabetes_response_training, diabetes_response_testing) = \
        ms.train_test_split(predictors, response, test_size=0.2)
        # random_state=random_state) #, random_state=0)

    algorithm = lm.LogisticRegression(max_iter=100000)
    # algorithm = es.RandomForestClassifier()
    model = algorithm.fit(diabetes_predictors_training, diabetes_response_training)
    predictions = model.predict(diabetes_predictors_testing)
    show_accuracy(diabetes_response_testing, predictions)
    # show_accuracy(diabetes_response_testing, all_zeros)

    # create_confusion_matrix(diabetes_response_testing, predictions)


# Cross-validation. Sklearn makes it almost too simple. It's hard to remember
# that it's performing many steps that we often perform manually.
def perform_logistic_regression_with_cross_validation(predictors, response):    
    algorithm = lm.LogisticRegression(max_iter=100000)
    scores = \
        ms.cross_val_score(algorithm, predictors, response, scoring='accuracy', cv=10)
    print(f"Scores: {scores}, average: {np.mean(scores)}")


def analyze(diabetes_df):
    # diabetes_df = ['3']
    diabetes_df = diabetes_df.drop(['DNR Order', 'Med Tech'], axis='columns')
    predictors = diabetes_df.drop('Outcome', axis='columns')
    response = diabetes_df['Outcome']
    # print(predictors.columns)

    # for balance_counter in range(2):
    #     if balance_counter == 0:
    #         print("Unbalanced:")
    #     else:
    #         print("Balanced:")
    #     perform_logistic_regression(predictors, response, balance_counter)

    perform_logistic_regression_with_cross_validation(predictors, response)


def main():
    df = read_diabetes()
    # print(df.head())
    analyze(df)


if __name__ == "__main__":
    main()
