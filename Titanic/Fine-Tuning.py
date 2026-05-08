import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

titanic_df = pd.read_csv('Titanic/Titanic-Dataset.csv')

titanic_df = titanic_df[
    ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
].dropna()

titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})

titanic_df = pd.get_dummies(titanic_df, columns=['Embarked'])

print(titanic_df.head())

X = titanic_df.drop('Survived', axis=1)
y = titanic_df['Survived']

print("\nCorrelation with Survived:")
print(titanic_df.corr(numeric_only=True)['Survived'])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1
)

model = RandomForestClassifier(random_state=1)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)

print("\nReal Values:")
print(y_test.to_numpy())

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, predictions))

matrix = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(matrix)

print("\nTrue Negatives:")
print(matrix[0][0])

print("\nTrue Positives:")
print(matrix[1][1])