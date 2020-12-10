import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('data.csv')
print(dataset)

# from sklearn.datasets import load_breast_cancer
# nCancer = load_breast_cancer()
# print(nCancer)

x = dataset.iloc[:,2:-1].values
y = dataset.iloc[:,1].values
# np.set_printoptions(threshold = np.inf)
print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25)


#logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix
logistic_model = LogisticRegression(random_state = 0)
logistic_model.fit(x_train, y_train)
y_predict =logistic_model.predict(x_test)
cm = confusion_matrix(y_test,y_predict)

print(classification_report(y_test,y_predict))



#K- nearest neighbours model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report , confusion_matrix
knn_model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_model.fit(x_train, y_train)
y_predict =knn_model.predict(x_test)
cm = confusion_matrix(y_test,y_predict)

from sklearn import metrics
print(classification_report(y_test,y_predict))

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = []
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 