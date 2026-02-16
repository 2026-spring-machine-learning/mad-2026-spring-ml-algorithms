import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    cars = pd.read_csv(r"C:\Users\nicky\OneDrive - Madison College\Machine Learning\mad-2026-spring-ml-algorithms\mac-learn-pub\cars.csv")
    predict(cars)

def predict(cars_df: pd.DataFrame) -> None:
    Engine_Size = cars_df[['enginesize']]
    Prices = cars_df['price'].to_numpy()

    algorithm = lm.LinearRegression()
    model = algorithm.fit(Engine_Size, Prices)
    Sizes = model.predict(Engine_Size)
    print(Prices)
    print(Sizes)
    
    r_squared = model.score(Engine_Size, Prices)
    print(f"r^2: {r_squared}")

    sns.regplot(x='enginesize', y='price', data=cars_df)
    plt.show()

if __name__ == "__main__":
    main()