# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn

```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### Data Head:
<img width="1777" height="302" alt="image" src="https://github.com/user-attachments/assets/6ca794cd-d42d-49fa-b613-108774cd2050" />


### Information:
<img width="1787" height="427" alt="image" src="https://github.com/user-attachments/assets/a9173a51-84b4-432f-a6d2-26e16746e4d5" />


### Null dataset:
<img width="1744" height="536" alt="image" src="https://github.com/user-attachments/assets/e45f9984-b945-4574-a1c7-63249a53905f" />


### Value_counts():
<img width="1767" height="265" alt="image" src="https://github.com/user-attachments/assets/770ed5fb-8492-41cd-a32a-4cf0eb12d516" />


### Data Head:
<img width="1776" height="300" alt="image" src="https://github.com/user-attachments/assets/1a9d7a48-c969-4576-92ca-737450fb1eb0" />


### x.head():
<img width="1660" height="293" alt="image" src="https://github.com/user-attachments/assets/03fed685-655a-4c9e-9694-3f62b6151a11" />


### Accuracy:
<img width="1152" height="86" alt="image" src="https://github.com/user-attachments/assets/8606320f-1fd7-4275-b321-1839f5f8b45b" />


### Data Prediction:
<img width="1774" height="130" alt="image" src="https://github.com/user-attachments/assets/ec167853-54d5-4465-a42b-be25da261eb3" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
