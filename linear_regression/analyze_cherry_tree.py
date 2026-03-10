import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.metrics as metrics

# Add plotting imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns


def feature_select(tree_df):
    one_hot_encoded_columns = pd.get_dummies(tree_df[['Season']])
    # Won't work with Season as strings!
    # predictors_df = tree_df[['Height', 'Diam', 'Season']]
    predictors_df = tree_df[['Height', 'Diam']]
    # predictors_df = pd.concat([predictors_df, one_hot_encoded_columns], axis='columns')

    # Most of this block is for linearity.
    predictors_and_response_df = pd.concat([predictors_df, tree_df['Volume']], axis='columns')
    print(predictors_and_response_df)
    correlation_matrix = predictors_and_response_df.corr()
    volume_correlation_matrix = correlation_matrix[['Volume']].sort_values(by='Volume', ascending=False)
    print(volume_correlation_matrix)
    # print(correlation_matrix)
    # sns.heatmap(volume_correlation_matrix, annot=True)
    # plt.show()

    # For independence.
    print(correlation_matrix)
    mask = abs(correlation_matrix < 0.3)
    # sns.heatmap(correlation_matrix, annot=True, mask=mask)
    # plt.show()

    response_series = tree_df['Volume'].to_numpy()

    return (predictors_df, response_series)


def predict(cherry_tree_df):
    print(cherry_tree_df)
    (predictors_df, response_series) = feature_select(cherry_tree_df)

    # print("Before print")
    # print(predictors_df)
    # print(response_series)
    # print("After print")

    (predictors_training_df, predictors_testing_df, \
        response_training_df, response_testing_df) \
            = ms.train_test_split(predictors_df, response_series, test_size=0.2)

    # Train and predict.
    algorithm = lm.LinearRegression()
    model = algorithm.fit(predictors_training_df, response_training_df)
    prediction = model.predict(predictors_testing_df)
    print("Prediction vs Actual:")
    print(prediction)
    print(response_testing_df)

    mse = metrics.mean_squared_error(response_testing_df, prediction)
    rmse = np.sqrt(mse)
    print(f"rmse: {rmse}.")

    # Calculate r-squared.
    r_squared = model.score(predictors_df, response_series)
    print(f"r^2: {r_squared}")

    # Plot: Height (x) vs Volume (y)
    height = cherry_tree_df["Height"].to_numpy()
    volume = cherry_tree_df["Volume"].to_numpy()

    # Best-fit line (simple linear regression: Volume ~ Height)
    slope, intercept = np.polyfit(height, volume, 1)
    height_line = np.linspace(height.min(), height.max(), 200)
    volume_line = slope * height_line + intercept

    plt.figure(figsize=(9, 7))
    plt.scatter(height, volume, s=40)
    plt.plot(height_line, volume_line)
    plt.xlabel("Height")
    plt.ylabel("Volume")
    plt.title("Cherry Tree Volume vs Height")
    plt.tight_layout()
    plt.show()


def main():
    cherry_tree_df = pd.read_csv("CherryTree.csv")
    predict(cherry_tree_df)


if __name__ == "__main__":
    main()
