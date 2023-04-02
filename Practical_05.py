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

import statistics
from matplotlib import scale
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

GaussianNBmodel = GaussianNB()
clf_entropy = DecisionTreeClassifier()
neigh = KNeighborsClassifier()
scale = StandardScaler()


def get_Score(model,X_train,Y_train,X_test,Y_test):
    model.fit(X_train,Y_train)
    return model.score(X_test,Y_test)*100

for k in range(2):
    if (k == 0):
        print("Data Set : Abalone.")
        data = pd.read_csv('abalone.csv')
        X = data.values[:, 1:9]
        Y = data.values[:, 0]
    else:
        print("Data Set : Breast-Cancer.")
        data = pd.read_csv('breast-cancer.csv', na_values=['?'])
        data.dropna(inplace=True)
        X = data.values[:, 1:10]
        Y = data.values[:, 0]

    # Using 75% training set
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25)

    clf_entropy.fit(X_train, Y_train)
    neigh.fit(X_train, Y_train)
    GaussianNBmodel.fit(X_train, Y_train)

    y_pred_en = clf_entropy.predict(X_test)

    print(("Accuracy is "), accuracy_score(Y_test, y_pred_en)
            * 100, ("when using Decision Tree with 75 % of training data"))
    print(("Accuracy is "), neigh.score(X_test, Y_test)
            * 100, ("when using KNN with 75 % of training data"))
    print(("Accuracy is "), GaussianNBmodel.score(X_test, Y_test)
            * 100, ("when using Naive bayes with 75 % of training data"))        

    # Using 66.6 % training set

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.33)

    clf_entropy.fit(X_train, Y_train)
    neigh.fit(X_train, Y_train)
    GaussianNBmodel.fit(X_train, Y_train)

    y_pred_en = clf_entropy.predict(X_test)

    print(("Accuracy is "), accuracy_score(Y_test, y_pred_en)
            * 100, ("when using Decision Tree with 66.6 % of training data"))
    print(("Accuracy is "), neigh.score(X_test, Y_test)
            * 100, ("when using KNN with 66.6 % of training data"))
    print(("Accuracy is "), GaussianNBmodel.score(X_test, Y_test)
            * 100, ("when using Naive bayes with 66.6 % of training data"))     


data = pd.read_csv('abalone.csv')
X = data.values[:, 1:9]
Y = data.values[:, 0]
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
print("Standarized Data \n",X)
print("This is only possible for Abalone Dataset due to nature of Data.")


Accuracy = []
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1000,shuffle=True,stratify=Y)
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)
Accuracy.append(get_Score(neigh,X_train,Y_train,X_test,Y_test))

# using Random Subsampling for splitting
Accuracy_Random=[]
k=6
for i in range(0,k):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1000,shuffle=True,stratify=Y)
    X_train = scale.fit_transform(X_train)
    X_test = scale.fit_transform(X_test)
    Accuracy_Random.append(get_Score(neigh,X_train,Y_train,X_test,Y_test))
Accuracy.append(statistics.mean(Accuracy_Random))

# using K-Cross-Validation for splitting
k=9
kf = StratifiedKFold(n_splits=k)
Accuracy_kFold=[]
for train_index,test_index in kf.split(X,Y):
    X_train,X_test,Y_train,Y_test = X[train_index],X[test_index],Y[train_index],Y[test_index] # type: ignore
    X_train = scale.fit_transform(X_train)
    X_test = scale.fit_transform(X_test)
    Accuracy_kFold.append(get_Score(neigh,X_train,Y_train,X_test,Y_test))
Accuracy.append(statistics.mean(Accuracy_kFold))
print("Accuracy: ",Accuracy)

# Visualizing the accuracy of the K-Nearest Neighbour Model for different Splitting models
Yval = Accuracy
Xval=["Hold-Out","Random Sub-Sampling","Cross-Validation"]
plt.bar(Xval,Yval,color="green",width=0.2)
plt.xlabel("Splitting Method")
plt.title("K-Nearest Neighbor Classifier Visualization")
plt.show()