
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt 

cars_df = pd.read_csv('linear_regression/cars.csv')
#print(cars_df.head())

def prepare_data(cars_df):
    encoded_cols = pd.get_dummies(cars_df['carbody'])
    cars_df = pd.concat([encoded_cols, cars_df], axis=1)
    columns_to_keep = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 
        'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 
        'horsepower', 'peakrpm', 'citympg', 'highwaympg', 
        'convertible', 'hardtop', 'hatchback', 'sedan', 'wagon', 
        'price']
    cars_df = cars_df[columns_to_keep]
    return cars_df

def test_linearity(df):
    correlations = df.corr()
    price_corr = correlations[['price']].sort_values(by='price', ascending=False)
    print(price_corr)
    
    strong_predictors = price_corr[abs(price_corr['price']) > 0.5]
    print(strong_predictors)
    
    plt.figure(figsize=(6, 8))
    sns.heatmap(price_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.show()

prepared_df = prepare_data(cars_df)
test_linearity(prepared_df)

predictors_df = prepared_df.drop(columns=['price'])
response = prepared_df['price']

lr_model = LinearRegression()
lr_model.fit(predictors_df, response)
y_pred = lr_model.predict(predictors_df)
r_squared = lr_model.score(predictors_df, response)
print(f"R-squared: {r_squared}")

sns.lmplot(x='enginesize', y='price', data=cars_df)
plt.show()