# Question 1
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
%matplotlib inline
from sklearn.datasets import load_iris

# Question 2
#(a)
iris=load_iris()
x=iris.data
y=iris.target
x.shape,y.shape

#(b)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
from sklearn.naive_bayes import GaussianNB

#(c)
model=GaussianNB()
model.fit(x_train,y_train)
pred = model.predict(x_test)

#(c(i))
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))

#(c(ii))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred)
from sklearn.naive_bayes import MultinomialNB

#(d)
model=MultinomialNB()
model.fit(x_train,y_train)
pred = model.predict(x_test)

#(d(i))
print(classification_report(y_test,pred))

#(d(ii))
confusion_matrix(y_test,pred)

