import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



dataset = pd.read_csv('data.csv')
print('Dataset dimensions :', dataset.shape)
print()
print(dataset.head(10))
print()
print(dataset.isna().sum())
print()
dataset = dataset.dropna(axis=1)
print(dataset.head(10))
print()

print(dataset['diagnosis'].value_counts())
print()
# sns.countplot(dataset['diagnosis'],label="Count")
# plt.show()

#Visualization per Pair
# sns.set(style="ticks", color_codes=True)
# sns.pairplot(dataset.iloc[:,1:11],palette=('b','r'), hue = "diagnosis", height=2.5)
# plt.show()

from sklearn.preprocessing import LabelEncoder
labelenc = LabelEncoder()
dataset.iloc[:,1]= labelenc.fit_transform(dataset.iloc[:,1].values)
print(dataset.iloc[:,1].values)
print()

plt.figure(figsize=(16,16))
sns.heatmap(dataset.corr(), annot = True, fmt='.0%')
plt.show()

x = dataset.iloc[:,2:-1].values
y = dataset.iloc[:,1].values
# print(x)
# print()
# print(y)
# print()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logistic_model = LogisticRegression(random_state = 0)
logistic_model.fit(x_train, y_train)

y_predict = logistic_model.predict(x_test)

print('Logistic Regression Model :')
print()
print(metrics.confusion_matrix(y_test,y_predict))
print(metrics.classification_report(y_test,y_predict))
print('Accuracy :', metrics.accuracy_score(y_test,y_predict))
print()



#K- nearest neighbours model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

Ks = 10
mean_acc = np.zeros((Ks-1))
confusionMat = []
classificationRpt = []

for n in range(1,Ks): 
    knn_model = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    y_predict = knn_model.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, y_predict)

    # print(metrics.accuracy_score(y_test, y_predict))
    # print(metrics.confusion_matrix(y_test,y_predict))
    confusionMat.append(metrics.confusion_matrix(y_test,y_predict))
    # print(metrics.classification_report(y_test,y_predict))
    classificationRpt.append(metrics.classification_report(y_test,y_predict))

print('KNN Model :')
print()
print(confusionMat[mean_acc.argmax()])
print(classificationRpt[mean_acc.argmax()])
print("The best accuracy was", mean_acc.max(), "with k=", mean_acc.argmax()+1)



#Using SVM linear
from sklearn.svm import SVC
from sklearn import metrics
svc_lin = SVC(kernel = 'linear', random_state = 0)
svc_lin.fit(x_train, y_train)

y_predict = svc_lin.predict(x_test)

print('SVM Linear :')
print()
print(metrics.confusion_matrix(y_test,y_predict))
print(metrics.classification_report(y_test,y_predict))
print('Accuracy :', metrics.accuracy_score(y_test,y_predict))
print()



#Using SVM rbf
from sklearn.svm import SVC
from sklearn import metrics
svc_rbf = SVC(kernel = 'rbf', random_state = 0)
svc_rbf.fit(x_train, y_train)

y_predict = svc_rbf.predict(x_test)

print('SVM RBF :')
print()
print(metrics.confusion_matrix(y_test,y_predict))
print(metrics.classification_report(y_test,y_predict))
print('Accuracy :', metrics.accuracy_score(y_test,y_predict))
print()



#Using DecisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
tree.fit(x_train, y_train)

y_predict = tree.predict(x_test)

print('Decision Tree :')
print()
print(metrics.confusion_matrix(y_test,y_predict))
print(metrics.classification_report(y_test,y_predict))
print('Accuracy :', metrics.accuracy_score(y_test,y_predict))
print()



#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(x_train, y_train)

y_predict = forest.predict(x_test)

print('Random Forest :')
print()
print(metrics.confusion_matrix(y_test,y_predict))
print(metrics.classification_report(y_test,y_predict))
print('Accuracy :', metrics.accuracy_score(y_test,y_predict))
print()