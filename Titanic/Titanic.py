import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

titanic_df = pd.read_csv('Titanic/Titanic-Dataset.csv')

titanic_df = titanic_df[
    ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
].dropna()

print(titanic_df.head())

X = titanic_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic_df['Survived']

print("\nCorrelation with Survived:")
print(titanic_df.corr(numeric_only=True)['Survived'])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)

print("\nReal Values:")
print(y_test.to_numpy())

logistic_accuracy = accuracy_score(y_test, predictions)

print("\nLogistic Regression Accuracy:")
print(logistic_accuracy)

all_ones = [1] * len(y_test)

all_ones_accuracy = accuracy_score(y_test, all_ones)

print("\nAll 1s Accuracy:")
print(all_ones_accuracy)