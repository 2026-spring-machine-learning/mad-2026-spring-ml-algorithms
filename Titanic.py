import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("Titanic-Dataset.csv")
cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
df = df[cols]

df = df.dropna()

#print(df.shape)
#print(df.head())

print(df.corr(numeric_only=True)["Survived"])

X = df[["Pclass", "Age", "SibSp", "Parch", "Fare"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Predictions:", pred[:20])
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

all_ones = np.ones_like(y_test)
print("All 1 accuracy:", accuracy_score(y_test, all_ones))
print("Actual y_test values:", y_test.values[:20])