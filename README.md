# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Evangelin.S
RegisterNumber:  212221230025
*/

import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer 
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

### Output:
## Data Head:
![image](https://user-images.githubusercontent.com/94219798/174030422-20482dc2-5d99-4210-b1d3-eff4f1162143.png)

## Data Info:
![image](https://user-images.githubusercontent.com/94219798/174030469-4c93e78c-feef-47ac-b17e-56eb967ed33f.png)

## Data isnull():
![image](https://user-images.githubusercontent.com/94219798/174030525-f358b7a2-1469-45ac-99e3-6258e688a93f.png)

## y_pred:
![image](https://user-images.githubusercontent.com/94219798/174030579-838ede20-04eb-41ea-aec7-101faf078e85.png)

## Accuracy:
![image](https://user-images.githubusercontent.com/94219798/174030634-807257e3-47fa-4d99-8919-afb8f3c3a80b.png)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
