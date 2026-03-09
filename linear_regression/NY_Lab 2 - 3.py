import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.linear_model as lm

def main():
    cars = pd.read_csv(r"C:\Users\nicky\OneDrive - Madison College\Machine Learning\mad-2026-spring-ml-algorithms\mac-learn-pub\cars.csv")
    predict(cars)

def predict(cars: pd.DataFrame) -> None:
    print(cars.head())

    one_hot_encoded_columns = pd.get_dummies(cars[['carbody']])
    cars = cars.drop(columns=['carbody'])
    cars = pd.concat([cars, one_hot_encoded_columns], axis=1)

    predictors_df = cars[['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']]

    predictors_df = pd.concat([predictors_df, one_hot_encoded_columns], axis=1)
    response_series = cars['price'].to_numpy()
    model = lm.LinearRegression().fit(predictors_df, response_series)
    prediction = model.predict(predictors_df)
    print(response_series)
    print(prediction)
    
    print(f"r^2: {model.score(predictors_df, response_series)}")

    sns.regplot(x='enginesize', y='price', data=cars)
    plt.show()


if __name__ == "__main__":
   main()