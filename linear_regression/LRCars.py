from urllib import response
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt 

cars_df = pd.read_csv('cars.csv')
#print(cars_df.head())

def predict(cars_df):
    predictors = cars_df[['enginesize']]
    #predictors = cars_df['enginesize']
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