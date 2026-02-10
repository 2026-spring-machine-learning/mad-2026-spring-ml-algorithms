import numpy as np
import pandas as pd
import sklearn.linear_model as lm

# Add plotting imports
import matplotlib.pyplot as plt
import seaborn as sns


def predict(cherry_tree_df: pd.DataFrame) -> None:
    # print(cherry_tree_df.head())
    # print(cherry_tree_df.describe(include="all"))

    predictors_df = cherry_tree_df[['Height']]
    response_series = cherry_tree_df['Volume'].to_numpy()

    # Train and predict.
    algorithm = lm.LinearRegression()
    model = algorithm.fit(predictors_df, response_series)
    prediction = model.predict(predictors_df)
    print(response_series)
    print(prediction)

    # Calculate r-squared.
    r_squared = model.score(predictors_df, response_series)
    print(f"r^2: {r_squared}")

    # Plot Height vs Volume with best fit line
    plt.figure(figsize=(8, 6))
    sns.regplot(x=cherry_tree_df['Height'], y=cherry_tree_df['Volume'])
    plt.xlabel('Height')
    plt.ylabel('Volume')
    plt.title('Cherry Tree Height vs Volume with Best Fit Line')
    plt.tight_layout()
    plt.show()


def main():
    cherry_tree_df = pd.read_csv("CherryTree.csv")
    predict(cherry_tree_df)


if __name__ == "__main__":
    main()
