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

    # --- Part 2: One-Hot Encoding of carbody ---
    encoded = pd.get_dummies(cars_df['carbody'])
    cars_df = pd.concat([encoded, cars_df], axis=1)
    cars_df.drop(columns=['carbody'], inplace=True)

    # --- Part 3: Multiple Regression ---
    predictor_cols = [
        'symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight',
        'curbweight', 'enginesize', 'boreratio', 'stroke',
        'compressionratio', 'horsepower', 'peakrpm', 'citympg',
        'highwaympg', 'convertible', 'hardtop', 'hatchback', 'sedan',
        'wagon'
    ]

    cars_df = cars_df[predictor_cols + ['price']]

    predictors_df = cars_df[predictor_cols]
    response = cars_df['price']

    model = LinearRegression()
    model.fit(predictors_df, response)
    predictions = model.predict(predictors_df)
    r2 = model.score(predictors_df, response)
    print(f"Part 3 – Multiple Regression (19 predictors -> price)")
    print(f"  r² = {r2:.4f}\n")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(response, predictions, alpha=0.6)
    ax.plot([response.min(), response.max()], [response.min(), response.max()], 'r--', label='Perfect fit')
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title(f"Part 3 – Multiple Regression: Actual vs Predicted Price\nr² = {r2:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    cars_df = pd.read_csv('linear_regression/cars.csv')
    predict(cars_df)
