# XGBoost

# Install xgboost following the instructions on this link: http://xgboost.readthedocs.io/en/latest/build.html#


#Part 1: Data preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3 : 13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() #for  countries
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])# column [1] for countries
labelencoder_X_2 = LabelEncoder() #for gender
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])# column [1] for gender
onehotencoder = OneHotEncoder(categorical_features = [1])# create dummy variable for countries column
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:] #remove the 1st column to avoid dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting the XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10 )
                #cv: nb of fold we want to create
#if we work on the large dataset, we need to set the n_jobs that is for running all the CPUs
accuracies.mean()
accuracies.std()