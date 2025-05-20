# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.
2.Analyse the data.
3.Use modelselection and Countvectorizer to preditct the values.
4.Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MOHAMMED HAMZA M
RegisterNumber:  212224230167
*/


import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:

 ## data:
 ![image](https://github.com/user-attachments/assets/22cfdfc4-9000-4357-b42f-a5ccdf921cfc)

## data.shape():
![image](https://github.com/user-attachments/assets/a36cc1fa-2ce4-49d7-920c-d37512ed52a7)

## x.shape():
![image](https://github.com/user-attachments/assets/9d115a70-3187-49e5-9e57-728ecd6c9fcd)

## y.shape():
![image](https://github.com/user-attachments/assets/f52075c2-c864-4e9c-8818-a89a63b74558)

## x_train:
![image](https://github.com/user-attachments/assets/e2007a18-bcee-472f-9753-1439e781bcc2)

## x_train.shape()
![image](https://github.com/user-attachments/assets/3c6f8453-b531-4cbb-ada4-fc4f1ab36e78)

## y pred:
![image](https://github.com/user-attachments/assets/f9210316-9896-4273-ae7c-fb391f0c2188)

## acc(accuracy):
![image](https://github.com/user-attachments/assets/0c6b021c-6e64-4559-a76a-4e1e07e0b158)

## con(confusion matrix):
![image](https://github.com/user-attachments/assets/b7ce0d5e-d3de-4889-9a81-209b96ec9e9d)

## cl (classification report):
![image](https://github.com/user-attachments/assets/819aeb94-5d92-4ed3-9ee8-a4f6caa2f831)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
