import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def predict(cars_df):
    # --- Part 1: Simple Linear Regression (engine size -> price) ---
    predictors_df = cars_df[['enginesize']]
    response = cars_df['price']

    model = LinearRegression()
    model.fit(predictors_df, response)
    predictions = model.predict(predictors_df)
    r2 = model.score(predictors_df, response)
    print(f"Part 1 – Simple Linear Regression (enginesize -> price)")
    print(f"  r² = {r2:.4f}\n")

    sns.lmplot(x='enginesize', y='price', data=cars_df)
    plt.title("Engine Size vs Price")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    cars_df = pd.read_csv('linear_regression/cars.csv')
    predict(cars_df)
