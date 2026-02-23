import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import seaborn as sns

def predict(cars_df: pd.DataFrame) -> None:

    #part 2
    body_dummies = pd.get_dummies(cars_df['carbody'], prefix ='carbody')
    cars_df = pd.concat([cars_df, body_dummies], axis = 1)
    cars_df = cars_df.drop('carbody', axis=1)

    predictors_df = cars_df[['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg'] + list(body_dummies.columns)]
    response_series = cars_df['price'].to_numpy()
  

    #Train and predict.
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
    sns.regplot(x=cars_df['enginesize'], y=cars_df['price'])
    plt.xlabel('Engine Size')
    plt.ylabel('Price')
    plt.title('Engine Size vs Price with Best Fit Line')
    plt.tight_layout()
    plt.show()

def main():
    cars_df = pd.read_csv(r"C:\Users\kylas\Downloads\cars.csv")
    predict(cars_df)
    
if __name__ == "__main__":
    main()