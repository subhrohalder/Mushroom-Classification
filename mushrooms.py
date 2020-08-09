# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 21:24:07 2020

@author: subhrohalder
"""
#dataset link: https://www.kaggle.com/uciml/mushroom-classification
#imports
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
from  sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

#To Generate Report
def generate_model_report(y_actual,y_predicted):
    print('accuracy_score:',accuracy_score(y_actual,y_predicted))
    print('precision_score:',precision_score(y_actual,y_predicted))
    print('recall_score:',recall_score(y_actual,y_predicted))
    print('f1_score:',f1_score(y_actual,y_predicted))
    
#Reading the CSV
dataset=pd.read_csv('mushrooms.csv')
dataset.head()

#Data Selection 
y=dataset.loc[:,'class']
X=dataset.iloc[:,:]
X=X.drop(columns=['class'])

#One Hot encoding
X=pd.get_dummies(X,drop_first=True)
y=pd.get_dummies(y,drop_first=True)

#Split Train and Test set
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25)

#Model Fitting with complete imbalanced data
model=LogisticRegression()
model.fit(X_train,y_train)

#Prediction
y_pred= model.predict(X_test)

#accuracy report
generate_model_report(y_test,y_pred)
