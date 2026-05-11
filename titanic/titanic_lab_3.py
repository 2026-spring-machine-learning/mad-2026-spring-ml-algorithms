import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import sklearn.ensemble as ensemble
import matplotlib.pyplot as plt
import seaborn as sns
import imblearn.over_sampling as ios
import sklearn.metrics as metrics


def read_titanic():
    titanic=pd.read_csv(r"mad-2026-spring-ml-algorithms\titanic\Titanic-Dataset.csv")
    titanic=titanic.dropna()
    return titanic


def check_independence(predictors_df: pd.DataFrame, threshold: float = 0.8) -> None:
    predictor_correlation=predictors_df.corr()
    sns.heatmap(predictor_correlation, annot=True, mask=(predictor_correlation <= threshold))
    plt.show()


def check_linearity(titanic_df):
    plt.figure(figsize=(8, 6))
    sns.heatmap(titanic_df.corr()[['Survived']], annot=True)
    plt.show()


def compute_confusion_matrix_numbers(actual_data_df, prediction):
    confusion_tuple=metrics.confusion_matrix(actual_data_df, prediction)
    command_line_display_as_accuracy_top_confusion_matrix=confusion_tuple.T
    command_line_display_as_accuracy_top_confusion_matrix=np.flip(command_line_display_as_accuracy_top_confusion_matrix, axis=0)
    command_line_display_as_accuracy_top_confusion_matrix=np.flip(command_line_display_as_accuracy_top_confusion_matrix, axis=1)
    true_negs=0
    false_poss=0
    false_negs=0
    true_poss=0
    sensitivity=0
    specificity=0
    if len(confusion_tuple.ravel()) == 4:
        (true_negs, false_poss, false_negs, true_poss)=confusion_tuple.ravel()
        if ((true_poss + false_negs) > 0) and ((true_negs + false_poss) > 0):
            sensitivity=true_poss / (true_poss + false_negs)
            specificity=true_negs / (true_negs + false_poss)
            
    return confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs, false_poss, false_negs, true_poss, sensitivity, specificity

def create_confusion_matrix(actual_data_df, prediction):
    (confusion_tuple, accuracy, true_negs, false_poss, false_negs, true_poss, sensitivity, specificity)=compute_confusion_matrix_numbers(actual_data_df, prediction)
    print("TP:", true_poss,
          "TN:", true_negs,
          "FP:", false_poss,
          "FN:", false_negs)
    print("Accuracy:", accuracy)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    sns.heatmap(confusion_tuple, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def perform_random_forest(predictors, response, balance_counter):
    for random_state in range(0, 3):
        if balance_counter == 1:
            random_over_sampler=ios.RandomOverSampler(
                random_state=random_state)
            predictors, response=random_over_sampler.fit_resample(predictors, response)
    (titanic_predictors_training, titanic_predictors_testing, titanic_response_training, titanic_response_testing)=ms.train_test_split(
        predictors, response, test_size=0.2, random_state=1)
    algorithm=ensemble.RandomForestClassifier(
        n_estimators=100, random_state=1)

    model=algorithm.fit(
        titanic_predictors_training, titanic_response_training)
    predictions=model.predict(titanic_predictors_testing)
    print('Predictions:')
    print(predictions)
    print('Comparison:')
    print(predictions == titanic_response_testing.values)
    print('Accuracy:')
    print((predictions == titanic_response_testing.values).mean())
    
    return titanic_response_testing, predictions


def analyze(titanic_df):
    titanic_df=titanic_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked', 'Survived']]
    titanic_df['Sex']=titanic_df['Sex'].map({'male': 0, 'female': 1})
    titanic_df=pd.get_dummies(titanic_df,columns=['Embarked'])
    print(titanic_df)
    predictors=titanic_df.drop(columns='Survived')
    response=titanic_df['Survived']
    for balance_counter in range(2):
        if balance_counter == 0:
            print("Unbalanced:")
        else:
            print("Balanced:")
        actual, prediction=perform_random_forest(predictors, response, balance_counter)
        create_confusion_matrix(actual, prediction)


def main():
    titanic=read_titanic()
    analyze(titanic)


if __name__ == "__main__":
    main()
