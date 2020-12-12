import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('data.csv')
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

from sklearn.preprocessing import LabelEncoder
labelenc = LabelEncoder()
dataset.iloc[:,1]= labelenc.fit_transform(dataset.iloc[:,1].values)
print(dataset.iloc[:,1].values)
print()

# plt.figure(figsize=(20,20))
# sns.heatmap(dataset.corr(), annot = True, fmt='.0%')
# plt.show()

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
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
logistic_model = LogisticRegression(random_state = 0)
logistic_model.fit(x_train, y_train)

y_predict = logistic_model.predict(x_test)
cm = confusion_matrix(y_test,y_predict)

print('Logistic Regression Model :')
print(classification_report(y_test,y_predict))
print('Accuracy :', accuracy_score(y_test,logistic_model.predict(x_test)))
print()



#K- nearest neighbours model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

Ks = 10
mean_acc = np.zeros((Ks-1))
classificationRpt = []

for n in range(1,Ks): 
    knn_model = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    y_predict = knn_model.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, y_predict)

    # print(metrics.classification_report(y_test,y_predict))
    classificationRpt.append(metrics.classification_report(y_test,y_predict))

print('KNN Model :')
print(classificationRpt[mean_acc.argmax()])
print("The best accuracy was", mean_acc.max(), "with k=", mean_acc.argmax()+1)



# #Using SVM Model
