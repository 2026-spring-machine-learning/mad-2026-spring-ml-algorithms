import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

#load data
def load_data():
    past_data = pd.read_csv('Model/sole_survivor_past.csv')
    next_data = pd.read_csv('Model/sole_survivor_next.csv')
    return past_data, next_data

#split features and target
def split_features_target(df):
    X = df.drop(columns=['Name', 'SurvivalScore'])
    y = df['SurvivalScore']
    return X, y

#train model
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

#evaluate model
def evaluate_model(model, X, y):
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"R^2: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    return preds

#predict next season
def predict_next(model, next_data):
    X_next = next_data.drop(columns=['Name'])
    return model.predict(X_next)

#get top 3
def get_top_3(next_data, preds):
    results = next_data.copy()
    results['PredictedScore'] = preds
    top3 = results.sort_values('PredictedScore', ascending=False).head(3)
    return top3[['Name', 'PredictedScore']]

#graph
def plot_results(y, preds):
    plt.scatter(y, preds)
    plt.xlabel('Actual Survival Score')
    plt.ylabel('Predicted Survival Score')
    plt.title('Actual vs Predicted Survival Scores')
    plt.show()

def main():
    past_data, next_data = load_data()
    #print("Correlation with survival score:")
    #print(past_data.drop(columns=['Name']).corr()['SurvivalScore'].sort_values(ascending=False))
    X, y = split_features_target(past_data)
    model = train_model(X, y)
    preds = evaluate_model(model, X, y)
    plot_results(y, preds)
    next_preds = predict_next(model, next_data)
    top3 = get_top_3(next_data, next_preds)
    print("/nTop 3 contestants predicted to survive:")
    print(top3)

if __name__ == "__main__":
    main()