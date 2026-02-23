import numpy as np
import pandas as pd
import sklearn.linear_model as lm


# Add plotting imports
import matplotlib.pyplot as plt
import seaborn as sns



def predict(cars_df: pd.DataFrame) -> None:
    # print(cars_df.head())
    # print(cars_df.describe(include="all"))

    one_hot_encoded_columns = pd.get_dummies(cars_df[['carbody']])
    cars_df = cars_df.drop(columns=['carbody'])
    predictors_df = cars_df[['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']]
    predictors_df = pd.concat([predictors_df, one_hot_encoded_columns], axis='columns')
    response_series = cars_df['price'].to_numpy()

    # Train and predict.
    algorithm = lm.LinearRegression()
    model = algorithm.fit(predictors_df, response_series)
    prediction = model.predict(predictors_df)
    print(response_series)
    print(prediction)

    # Calculate r-squared.
    r_squared = model.score(predictors_df, response_series)
    print(f"r^2: {r_squared}")

    # Plot enginesize vs price with best fit line
    plt.figure(figsize=(8, 6))
    sns.regplot(x=cars_df['enginesize'], y=cars_df['price'])
    plt.xlabel('enginesize')
    plt.ylabel('price')
    plt.title('Car Engine Size vs Price with Best Fit Line')
    plt.tight_layout()
    plt.show()


def main():
    cars_df = pd.read_csv("H:/MATC/3_2026 Spring Sem/MachineLearning/labackup/cars.csv")
    predict(cars_df)
    print(cars_df)

if __name__ == "__main__":
    main()
