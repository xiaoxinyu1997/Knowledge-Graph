from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pickle

data = pd.read_csv('train_for_leaf.csv')
data.head()
X = data.iloc[:, :-98]
Y = data.iloc[:, -98:]

X_train, X_test, Y_train ,Y_test = train_test_split(X, Y, test_size=0.1)
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
s = model.score(X_test, Y_test)
print('Accuracy of Constraint value: ', s)
with open('leaf_model.pickle', 'wb') as sf:
    pickle.dump(model, sf)