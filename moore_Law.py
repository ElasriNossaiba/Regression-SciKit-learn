
import re #to handle csv files
import numpy as np
import matplotlib.pyplot as plt


X=[] #empty list for dependent variables
Y=[] #empty list for independent variable
#regex to remove any non decimal characters from the input
non_decimal = re.compile(r'[^\d]+')
for line in open('moore.csv'):
    r=line.split('\t') # separate data by a tab
# store the Year in X and delete wikipedia references
    x=int(non_decimal.sub('',r[2].split('[')[0]))
# transistor count is Y
    y=int(non_decimal.sub('',r[1].split('[')[0]))
    X.append(x)
    Y.append(y)
#plot the data
#converte to numpy array because they're easier to work with
X = np.array(X)
Y= np.array(Y)
plt.scatter(X,Y)
plt.show()
#log transforme data 
Y=np.log(Y)
plt.scatter(X,Y)
plt.show()
# our model
denominator = X.dot(X) - X.mean() * X.sum()
a= (X.dot(Y) - Y.mean()*X.sum())/denominator
b= (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator
Yhat = a*X+b
plt.scatter(X,Y)
plt.plot(X,Yhat)
plt.show()
# R-squared to evaluate our model
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)
print("a:",a,"b:",b)
print("the r-squared is: ",r2)

#log(tc) = a*year + b
#tc = exp(a*year)*exp(b)
#2 * tc = 2 * exp(a*year)*exp(b) = exp(ln(2)) * exp(b) * exp(a*year)
#                                = exp(b) * exp(a*year + ln(2))
# exp(a*year2)*exp(b)=exp(b) * exp(a*year1 + ln(2))
# a*year2 = a*year1 + ln(2)
# year2 = year1 + ln(2)/a
print("time to double:", np.log(2)/a, "years") 