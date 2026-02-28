import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.linear_model as lm

def predict(cars_df: pd.DataFrame) -> None:
    print(cars_df.head())

    one_hot_encoded_columns = pd.get_dummies(cars_df[['carbody']])
    cars_df = cars_df.drop(columns=['carbody'])
    cars_df = pd.concat([cars_df, one_hot_encoded_columns], axis=1)

    predictors_df = cars_df[['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']]

    predictors_df = pd.concat([predictors_df, one_hot_encoded_columns], axis=1)


    response_series = cars_df['price'].to_numpy()

    
    model = lm.LinearRegression().fit(predictors_df, response_series)
    prediction = model.predict(predictors_df)

    print(response_series)
    print(prediction)

    
    print(f"r^2: {model.score(predictors_df, response_series)}")

    
    plt.figure(figsize=(8, 6))
    sns.regplot(x=cars_df['enginesize'], y=cars_df['price'])
    plt.xlabel('enginesize')
    plt.ylabel('price')
    plt.title('Car Engine Size vs Price with Best Fit Line')
    plt.tight_layout()
    plt.show()

def main():
    cars_df = pd.read_csv('cars.csv')
    predict(cars_df)

if __name__ == "__main__":
   main()