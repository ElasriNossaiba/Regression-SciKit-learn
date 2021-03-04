
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nossaiba
"""
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt 

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

print(diabetes.feature_names)
x=diabetes.data[:,0:9]
y=diabetes.target

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=0) 

##Visualising the training set results
plt.scatter(X_train, y_train, color ='r')# plot data



model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

R1=model.score(X_train,y_train)
R2=model.score(X_test,y_test)


print('\nPR2 train: %0.1f' %R1)
print('\nPR2 test: %0.1f' %R2)


##Visualising the training set results
plt.plot(X_train, model.predict(X_train), color='b') # plot our model
plt.show()
#
##Visualising the Test set results
plt.scatter(X_test, y_test, color ='g') 
plt.plot(X_test, model.predict(X_test), color='b')
plt.show()
