
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt 

cars_df = pd.read_csv('linear_regression/cars.csv')
#print(cars_df.head())

def predict(cars_df):
    encoded_cols = pd.get_dummies(cars_df['carbody'])
    cars_df = pd.concat([encoded_cols, cars_df], axis=1)
    columns_to_keep = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 
        'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 
        'horsepower', 'peakrpm', 'citympg', 'highwaympg', 
        'convertible', 'hardtop', 'hatchback', 'sedan', 'wagon', 
        'price']
    cars_df = cars_df[columns_to_keep]
    predictors = cars_df.drop(columns=['price'])
    return predictors

predictors_df = predict(cars_df)
response = cars_df['price']
#print(predictors_df.head())

lr_model = LinearRegression()
lr_model.fit(predictors_df, response)
y_pred = lr_model.predict(predictors_df)
#print(y_pred)
r_squared = lr_model.score(predictors_df, response)
print(f"R-squared: {r_squared}")

sns.lmplot(x='enginesize', y='price', data=cars_df)
plt.show()