# Use Naive bayes, K-nearest, and Decision tree classification algorithms and build classifiers.
# Divide the data set into training and test set. Compare the accuracy of the different classifiers under the following situations:
# 5.1 a) Training set = 75% Test set = 25% b) Training set = 66.6% (2/3rd of total), Test set = 33.3%
# 5.2 Training set is chosen by i) hold out method ii) Random subsampling iii) Cross-Validation.Compare the accuracy of the classifiers obtained.
# 5.3 Data is scaled to standard format.

# Following Modifications are made to the Original Data

# rangeValue -> Middle Value.
# premeno -> 1 , ge40 -> 2 , lt40-> 3
# no -> 0 , yes -> 1
# left -> 1 , right -> 2
# left_low -> 1 , right_up -> 2 , left_up -> 3 , right_low -> 4 , central -> 5
# no-recurrence-events -> N , recurrence-events -> R

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.impute import KNNImputer

for k in range(2):
    if (k == 0):
        print("Decision Tree : Abalone.")
        data = pd.read_csv('abalone.csv')
        X = data.values[:, 1:9]
        Y = data.values[:, 0]
    else:
        print("Decision Tree : Breast-Cancer.")
        data = pd.read_csv('breast-cancer.csv' , na_values=['?'])
        # imputer = KNNImputer(n_neighbors=2, weights="uniform")
        # imputer.fit_transform(data)
        data.dropna(inplace=True)
        X = data.values[:, 1:10]
        Y = data.values[:, 0]

    # Decision Tree 75% training set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

    clf_entropy = DecisionTreeClassifier()
    clf_entropy.fit(X_train, Y_train)

    y_pred_en = clf_entropy.predict(X_test)

    print(("Accuracy is "), accuracy_score(Y_test, y_pred_en)
          * 100, ("with 75 % of training data"))

    # Decision Tree 66.6 % training set

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

    clf_entropy = DecisionTreeClassifier()
    clf_entropy.fit(X_train, Y_train)

    y_pred_en = clf_entropy.predict(X_test)

    print(("Accuracy is "), accuracy_score(Y_test, y_pred_en)
          * 100, ("with 66.6 % of training data"))
