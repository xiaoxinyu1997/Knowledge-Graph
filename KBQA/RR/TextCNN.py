import numpy as np
import tensorflow as tf
import random, os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

# len(X) = 1668 len(y) = 19

class CnnOrc:
     def cnnNet(self):
        weight = {
            # 输入 42*40*2
        }


data = pd.read_csv('train_for_relations.csv')
X = data.iloc[0, :1668].append(data.iloc[0, :12])
Y = data.iloc[0, 1668:]
# print(len(X))
# print(len(Y))
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=22)
X = tf.reshape(X, (1, 42, 40, 1))
# print(X)

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                               padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='sigmoid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])

for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, X.shape)

batch_size = 64
