from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pickle

data = pd.read_csv('train_for_relation.csv')
data.head()
X = data.iloc[:, :-19]
Y = data.iloc[:, -19:]

X_train, X_test, Y_train ,Y_test = train_test_split(X, Y, test_size=0.1)
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
s = model.score(X_test, Y_test)
print('Accuracy of Property name: ', s)
with open('relations_model.pickle', 'wb') as sf:
    pickle.dump(model, sf)