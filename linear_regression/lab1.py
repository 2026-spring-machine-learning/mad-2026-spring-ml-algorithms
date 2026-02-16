import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import seaborn as sns

#print(cars_df.head())

def predict(cars_df: pd.DataFrame) -> None:
    #print(cars_df.head())

    predictors_df = cars_df[['enginesize']]
    response_series = cars_df['price'].to_numpy()

    algorithm = lm.LinearRegression()
    model = algorithm.fit(predictors_df, response_series)
    prediction = model.predict(predictors_df)
    print(response_series)
    print(prediction)

    r_squared = model.score(predictors_df, response_series)
    print(f"r^2: {r_squared}")

    plt.figure(figsize=(8, 6)) 
    sns.regplot(x=cars_df['enginesize'], 
    y=cars_df['price']) 
    plt.xlabel('EngineSize') 
    plt.ylabel('Price') 
    plt.title('Engine Size vs Price with Best Fit Line') 
    plt.tight_layout() 
    plt.show()

def main():
    cars_df = pd.read_csv('cars.csv')
    predict(cars_df)

if __name__ == "__main__":
    main()