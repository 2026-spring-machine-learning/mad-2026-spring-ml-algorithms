import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import imblearn.over_sampling as ios
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

##This function tests correlation between the indepedent variables. It is called by feature_select
## No strong correlation between predictors, so we will keep all predictors for the model (except Pclass which is dropped in feature_select() due to low correlation with the response variable 'Survived')
def testing_ind(predictors_df):
    correlation_matrix = predictors_df.corr()
    print(f'This is the correlation matrix \n {correlation_matrix}')
    cor_mask = correlation_matrix <= 0.8
    
    sns.heatmap(correlation_matrix, mask=cor_mask, annot=True)
    plt.title('Correlation Matrix of Predictors for Independence Testing')
    plt.show()

    return predictors_df


##This function is called by predict and is used to determine which independent variables to use in the model. It also checks for linearity and independence of the predictors.
    # Independent Variables avail = 'Pclass', 'Age', 'SibSp', 'Fare'
    ## dropping Pclass due to being -0.035 below the 0.1 threshold for correlation with the response variable 'Survived'
    # Dependant Variable 'Survived'
def feature_select(df: pd.DataFrame) -> None:

    predictors = df[['Age', 'SibSp', 'Fare']]
    response = df['Survived']


    predictors_and_response_df = pd.concat([predictors, response], axis = 'columns')
    print(predictors_and_response_df)


    correlation_matrix = predictors_and_response_df.corr()


    print(f"This is the correlation matrix \n {correlation_matrix}")
    volume_correlation_matrix = correlation_matrix[['Survived']].sort_values(by='Survived', ascending=False)

    sns.heatmap(volume_correlation_matrix, annot=True)
    plt.title('Correlation Matrix of Predictors and Response for Feature Selection')
    plt.show()

    # Test independence between predictors
    testing_ind(predictors)

    # note: not using the 'best-fit' model because we have multiple Independent Variables. 
    return (predictors, response)




def show_accuracy(df_response_testing, predictions):
    print(f"Accuracy: {(predictions == df_response_testing.values).mean()}")
    # print((predictions == df_response_testing.values).mean())
    comparison = pd.DataFrame({
        'Actual': df_response_testing.values,
        'Predicted': predictions
    })
    print(comparison.head(20))  # Show first 20 for brevity
    print(f"Accuracy: {(predictions == df_response_testing.values).mean()}")

def perform_logistic_regression(predictors, response, balance_counter):
    ## will run 6 times -- 3 random states for unbalanced and 3 random states for balanced
    for random_state in range(0, 3):
        if balance_counter == 1:
            random_over_sampler = ios.RandomOverSampler(random_state=random_state)
            predictors, response \
                = random_over_sampler.fit_resample(predictors, response)

        (df_predictors_training, df_predictors_testing,
        df_response_training, df_response_testing) = \
            ms.train_test_split(predictors, response, test_size=0.2,
            random_state=random_state) #, random_state=0)-


        algorithm = lm.LogisticRegression(max_iter=100000)
        model = algorithm.fit(df_predictors_training, df_response_training)
        predictions = model.predict(df_predictors_testing)

        show_accuracy(df_response_testing, predictions)

        return model, df_predictors_testing, predictions


def analyze(df):
    
    predictors, response = feature_select(df)
    

    ## Produces 0 and 1 which is then passed into perform_logistic_regression() as balance_counter
    for balance_counter in range(2):
        if balance_counter == 0:
            print("Unbalanced:")
        else:
            print("Balanced:")
        perform_logistic_regression(predictors, response, balance_counter)

        return predictors, response

## Reads the csv file and drops any rows with missing values
def read_and_clean_df():
     df = pd.read_csv("H:/MATC/3_2026 Spring Sem/MachineLearning/mad-2026-spring-ml-algorithms/classification/Titanic-Dataset.csv")
     df = df.dropna()
     print(df.head())
     return df


def main():
    df = read_and_clean_df()
    # print(df.head())
    analyze(df)



if __name__ == "__main__":
    main()
