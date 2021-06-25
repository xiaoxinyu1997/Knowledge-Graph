from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

data = pd.read_csv('train_for_relations.csv')
X = data.iloc[:, :1668]
Y = data.iloc[:, 1668:]

X_train, X_test, Y_train ,Y_test = train_test_split(X, Y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

s = model.score(X_test, Y_test)

print(s)