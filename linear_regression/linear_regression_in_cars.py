import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import seaborn as sns

def predictor_correlation(predictors_df: pd.DataFrame, threshold: float = 0.8) -> None:

    predictor_correlation = predictors_df.corr()
    sns.heatmap(predictor_correlation, annot=True, mask=(predictor_correlation <= threshold))
    plt.show()

def predict(cars_df: pd.DataFrame) -> None:

    #part 2
    body_dummies = pd.get_dummies(cars_df['carbody'], prefix ='carbody')
    cars_df = pd.concat([cars_df, body_dummies], axis = 1)
    cars_df = cars_df.drop('carbody', axis=1)

    numbers_df = cars_df[['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price'] + list(body_dummies.columns)]

    #drop columns
    numbers_df = numbers_df.drop(columns=['curbweight'])
    numbers_df = numbers_df.drop(columns=['citympg'])
    numbers_df = numbers_df.drop(columns=['horsepower'])
    numbers_df = numbers_df.drop(columns=['carlength'])
    predictors_df = numbers_df.drop(columns=['price'])
    
    response_series = cars_df['price'].to_numpy()
    predictor_correlation(predictors_df, threshold=0.8) 

    #Train and predict.
    algorithm = lm.LinearRegression()
    model = algorithm.fit(predictors_df, response_series)  
    print(numbers_df.corr()['price'])
    #prediction = model.predict(predictors_df)
    #print(response_series)
    #print(prediction)
    
    # Plot Height vs Volume with best fit line
 
    plt.figure(figsize=(8, 6))
    sns.heatmap(numbers_df.corr()[['price']], annot=True)
    plt.show()
    sns.lmplot(x="enginesize", y="price", data=cars_df)
    plt.xlabel('Engine Size')
    plt.ylabel('Price')
    plt.title('Engine Size vs Price')
    plt.tight_layout()
    plt.show()

    # Calculate r-squared. 

    r_squared = model.score(predictors_df, response_series)
    print(f"r^2: {r_squared}")

    
def main():
    cars_df = pd.read_csv(r"C:\Users\kylas\Downloads\cars.csv")
    predict(cars_df)
    
if __name__ == "__main__":
    main()