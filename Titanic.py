import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

df = pd.read_csv("Titanic-Dataset.csv")
cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
df = df[cols]

df = df.dropna()

df["Sex"] = df["Sex"].map({"male" : 0, "female" : 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

#print(df.shape)
#print(df.head())

print(df.corr(numeric_only=True)["Survived"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.25, random_state=1
#)

#ros = RandomOverSampler(random_state=1)
#X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

#model = LogisticRegression(max_iter=100000)
#model.fit(X_train_res, y_train_res)
#pred = model.predict(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=1)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

#ros = RandomOverSampler(random_state=1)
#X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

#model.fit(X_train_res, y_train_res)
#pred = model.predict(X_test)

#print("Predictions:", pred[:20])
#print("Accuracy:", accuracy_score(y_test, pred))
#print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

#all_ones = np.ones_like(y_test)
#print("All 1 accuracy:", accuracy_score(y_test, all_ones))
#print("Actual y_test values:", y_test.values[:20])

print("10-fold accuracy scores:", scores)
print("Mean accuracy:", scores.mean())
print("Accuracy standard dev:", scores.std())