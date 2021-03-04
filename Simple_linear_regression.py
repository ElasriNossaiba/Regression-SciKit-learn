# -*- coding: utf-8 -*-
"""
Prediction du salaire d'un employé à partir de son expérience à l'aide de la régression linéaire simple
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd # import and manage datasets

#importing dataset
dataset = pd.read_csv('Salary_Data.csv')
#dataset = pd.read_csv('moore.csv')


X = dataset.iloc[:,:-1].values 
y = dataset.iloc[:,1].values 

#split data into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0) 


from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 
regressor.fit(X_train,y_train)  

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Visualising the training set results
plt.scatter(X_train, y_train, color ='red')# plot data
plt.plot(X_train, regressor.predict(X_train), color='blue') # plot our model
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#Visualising the Test set results
plt.scatter(X_test, y_test, color ='red') 
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



R1=regressor.score(X_train,y_train)
R2=regressor.score(X_test,y_test)
print('\nPR2 train: %0.1f' %R1)
print('\nPR2 test: %0.1f' %R2)


