import numpy as np
import pandas as pd
import sklearn.linear_model as lm

# Add plotting imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def predict(cherry_tree_df: pd.DataFrame) -> None:
    # print(cherry_tree_df.head())
    # print(cherry_tree_df.describe(include="all"))

    one_hot_encoded_columns = pd.get_dummies(cherry_tree_df[['Season']])
    # print(one_hot_encoded_columns)
    # Won't work with Season as strings!
    # predictors_df = cherry_tree_df[['Height', 'Diam', 'Season']]
    predictors_df = cherry_tree_df[['Height', 'Diam']]
    predictors_df = pd.concat([predictors_df, one_hot_encoded_columns], axis='columns')
    # print(predictors_df)
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

    # # 3D plot: Diameter and Height as independent variables, Volume as dependent.
    # fig = plt.figure(figsize=(9, 7))
    # ax = fig.add_subplot(111, projection="3d")

    # diam = cherry_tree_df["Diam"].to_numpy()
    # height = cherry_tree_df["Height"].to_numpy()
    # volume = cherry_tree_df["Volume"].to_numpy()

    # # Scatter of observed data
    # ax.scatter(diam, height, volume, s=40)

    # # Overlay the fitted regression plane
    # diam_grid = np.linspace(diam.min(), diam.max(), 25)
    # height_grid = np.linspace(height.min(), height.max(), 25)
    # diam_mesh, height_mesh = np.meshgrid(diam_grid, height_grid)

    # # Model was fit with predictors_df columns ['Height', 'Diam']
    # coef_height, coef_diam = model.coef_
    # volume_pred_mesh = model.intercept_ + coef_diam * diam_mesh + coef_height * height_mesh
    # # ax.plot_surface(diam_mesh, height_mesh, volume_pred_mesh, alpha=0.25, linewidth=0)

    # ax.set_xlabel("Diam")
    # ax.set_ylabel("Height")
    # ax.set_zlabel("Volume")
    # ax.set_title("Cherry Tree Volume vs Diameter and Height")
    # plt.tight_layout()
    # plt.show()


def main():
    cherry_tree_df = pd.read_csv("CherryTree.csv")
    predict(cherry_tree_df)


if __name__ == "__main__":
    main()
