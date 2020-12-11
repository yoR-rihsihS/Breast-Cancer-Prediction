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
# sns.heatmap(dataset.corr(), annot = True)
# plt.show()

x = dataset.iloc[:,2:-1].values
y = dataset.iloc[:,1].values
# print(x)
# print(y)

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

print(classification_report(y_test,y_predict))
print('Acc :', accuracy_score(y_test,logistic_model.predict(x_test)))
print()



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



# #Using SVM Model
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report , confusion_matrix
# from sklearn.svm import SVC
# svm_model = SVC(kernel = 'linear', random_state = 0)
# svm_model.fit(x_train, y_train)
# y_predict =svm_model.predict(x_test)
# cm = confusion_matrix(y_test,y_predict)

# # print(classification_report(y_test,y_predict))

# #model improvisation
# min_train =x_train.min()
# range_train =(x_train - min_train).max()
# x_train_scaled =(x_train-min_train)/range_train

# from sklearn.metrics import f1_score
# f1_score(y_test, yhat, average='weighted') 

# from sklearn.metrics import jaccard_similarity_score
# jaccard_similarity_score(y_test, yhat)