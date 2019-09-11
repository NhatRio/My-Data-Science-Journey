#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 20:06:10 2018

@author: PhuocNhatDANG
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('input/train.csv')
#train_set = train.values
test = pd.read_csv('input/test.csv')
#test_set = test.values
combine = [train, test]

train.head()

#Dropping the useless feature
train = train.drop(['PassengerId','Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)

train.describe()
test.describe()

train.info() 
test.info()

#See the role of gender for survived rate
train[["Sex", "Survived"]].groupby(['Sex'], \
        as_index=False).mean().sort_values(by='Survived', ascending=False)
#####Encoding categorical data######
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Sex = LabelEncoder()
train['Sex'] = labelencoder_Sex.fit_transform(train['Sex'])
test['Sex'] = labelencoder_Sex.transform(test['Sex'])

#Create a new feature Title
combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())#strip: remove whitespace
titles = dataset['Title'].unique()
    #Drop the column Name
train.drop(['Name'], axis = 1, inplace = True) 
test.drop(['Name'], axis = 1, inplace = True) 

#Find the mean of age for the missing age value
train['Age'].fillna(-1, inplace = True)
test['Age'].fillna(-1, inplace = True)
combine = [train]#, test]
medians_train = dict()
for title in titles:
    median = train.Age[(train["Age"] != -1) & (train['Title'] == title)].median()
    medians_train[title] = median

medians_test = dict()
for title in titles:
    median = test.Age[(test["Age"] != -1) & (test['Title'] == title)].median()
    medians_test[title] = median
medians_train['Dona'] = medians_test['Dona']
medians_test['Ms'] = medians_train['Ms']
from collections import Counter

sums = dict((Counter(medians_train) + Counter(medians_test)))
medians = {k: sums[k] / float((k in medians_train) + (k in medians_test)) for k in sums}


combine = [train, test]
for dataset in combine:
#    for index, row in dataset.iterrows():
#        if row['Age'] == -1:
#            dataset.loc[index, 'Age'] = medians[row['Title']]
    for title in titles:
        dataset['Age'].loc[ ( dataset['Age'] == -1) & (dataset['Title'] == title)]=medians[title]
        #dataset['Age'].loc[ ( dataset['Age'] == -1)] = dataset['Age'].median()
        
#See the role of Title for survived rate
train[["Title", "Survived"]].groupby(['Title'], \
        as_index=False).mean().sort_values(by='Survived', ascending=False)    

replacement_title={ 'Capt' : 0, 'Don' :0, 'Rev': 0, 'Jonkheer' :0, 'Dona' :0,
               'Mr' : 1,
               'Dr' : 2,
               'Major' : 3, 'Col' : 3,
               'Master' : 4,
               'Miss' : 5,
               'Mrs' : 6,
               'Mme' : 7, 'Lady':7, 'Ms' :7, 'Sir' :7, 'Mlle' :7, 'the Countess' :7 
              }
combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(replacement_title)
from sklearn.preprocessing import StandardScaler
sc_title = StandardScaler()
train['Title'] = sc_title.fit_transform(train['Title'].values.reshape(-1,1))
test['Title'] = sc_title.transform(test['Title'].values.reshape(-1,1))    

#Taking care of missing values
from sklearn.preprocessing import Imputer
# For Age
combine = [train, test]
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 14, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 14) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']=4

# Feature Scaling    
sc_age = StandardScaler()
train['Age'] = sc_age.fit_transform(train['Age'].values.reshape(-1,1))
test['Age'] = sc_age.transform(test['Age'].values.reshape(-1,1))    



    
# For  Fare (only on test)
test['Fare'].fillna(-1, inplace = True)
medians_test = dict()
for pclass in test['Pclass']:
    median = test.Fare[(test['Fare'] != -1) & (test['Pclass'] == pclass)].median()
    medians_test[pclass] = median


for pclass in test['Pclass']:
    test['Fare'].loc[ ( test['Fare'] == -1) & (test['Pclass'] == pclass)]=medians_test[pclass]

sc_fare = StandardScaler()
train['Fare'] = sc_fare.fit_transform(train['Fare'].values.reshape(-1,1))
test['Fare'] = sc_fare.transform(test['Fare'].values.reshape(-1,1))    
#For Embarked (only on train)
freq_port = train.Embarked.dropna().mode()[0] #mode: (most common value) 
                                                    #of discrete data.
                                                    #[0] avoid there are more 1 common
print(freq_port)

train['Embarked'] = train['Embarked'].fillna(freq_port)


train[['Embarked','Survived']].groupby(['Embarked'],\
          as_index = False).mean().sort_values(by='Survived', ascending = False)

replacement_embarked = {'S': 0, 'Q' : 1, 'C' : 2 }
train['Embarked'].replace(replacement_embarked, inplace = True)
test['Embarked'].replace(replacement_embarked, inplace = True)

sc_embarked = StandardScaler()
train['Embarked'] = sc_embarked.fit_transform(train['Embarked'].values.reshape(-1,1))
test['Embarked'] = sc_embarked.transform(test['Embarked'].values.reshape(-1,1))    

#Analyze by pivoting( the most important) features

train[['Pclass','Survived']].groupby(['Pclass'],\
          as_index = False).mean().sort_values(by='Survived', ascending = False)


train[["SibSp", "Survived"]].groupby(['SibSp'],\
        as_index=False).mean().sort_values(by='Survived', ascending=False)

train[["Parch", "Survived"]].groupby(['Parch'],\
        as_index=False).mean().sort_values(by='Survived', ascending=False)


train[['SibSp','Survived']].groupby('SibSp',\
     as_index=False).mean().sort_values(by = 'Survived', ascending = False )
replacement_SibSp={ 8: 0, 5 :0,
                    4 : 1,
                    3 : 2,
                    0 : 3, 
                    2 : 4,
                    1 : 5,
                }
combine = [train, test]
for dataset in combine:
    dataset['SibSp'] = dataset['SibSp'].replace(replacement_SibSp)
sc_SibSp = StandardScaler()
train['SibSp'] = sc_SibSp.fit_transform(train['SibSp'].values.reshape(-1,1))
test['SibSp'] = sc_SibSp.transform(test['SibSp'].values.reshape(-1,1))    
#For Parch
train[['Parch','Survived']].groupby('Parch',\
     as_index=False).mean().sort_values(by = 'Survived', ascending = False )
replacement_Parch={ 6: 0, 4 :0,
                    5 : 1,
                    0 : 2,
                    2 : 3, 
                    1 : 4,
                    3 : 5,
                }
combine = [train, test]
for dataset in combine:
    dataset['Parch'] = dataset['Parch'].replace(replacement_Parch)
sc_Parch = StandardScaler()
train['Parch'] = sc_Parch.fit_transform(train['Parch'].values.reshape(-1,1))
test['Parch'] = sc_Parch.transform(test['Parch'].values.reshape(-1,1))    

#Drop Cabin
combine = [train, test]
for dataset in combine:
    dataset.drop('Cabin', axis = 1, inplace = True)
    
sc_pclass = StandardScaler()
train['Pclass'] = sc_pclass.fit_transform(train['Pclass'].values.reshape(-1,1))
test['Pclass'] = sc_pclass.transform(test['Pclass'].values.reshape(-1,1))   




#Splitting the dataset into the Training set and the Test set 
X_train = train.drop(['Survived'],axis = 1)
Y_train = train['Survived']
X_test = test.drop(['PassengerId'], axis = 1)


#Part 2: Making the ANN

#Import the Keras libraries and packages
import keras 
from keras.models import Sequential  #initialize the neural network
from keras.layers import Dense # bulid the layers of ANN

#Initialising the ANN
classifier_ann = Sequential()

#Adding the input layer and the first hidden layer
classifier_ann.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8)) 
        # the 1st hidden layer has 6 nodes
        #'uniform': initialize the weights randomly and close to zero
        # 'relu': rectifier activation function
        ####### NOTE: rectifier for hidden layers and sigmoid func for output layer
# Adding the 2nd hidden layer
classifier_ann.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu')) 

        
# Adding the output layer
classifier_ann.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid')) 
        ####### NOTE: for the output more 2 categories (ex:3) we have to change the number
                # and the activation func by 3 and 'softmax'(softmax is the sigmoid
                    #function to three or more categories output)  
 #Compiling the ANN

classifier_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
        #optimizer: algorithm to find the optimal set of weights
        #loss: for the output more 2 categories (ex:3) categorical_crossentropy
        #metrics: criterion to evaluate our model
                                       
# Fitting the ANN to the Training set
classifier_ann.fit(X_train, Y_train, batch_size = 10, epochs = 200)
            #bacth_size: the number of observations after  which you want to update the weights
            #epochs : number of rounds that the whole training set pass through the ANN

#Making the predictions and evaluating the model
 
# Predicting the Test set results
Y_pred = classifier_ann.predict(X_test)
Y_pred = (Y_pred > 0.5)
#Y_pred = [int(elem) for elem in y_pred]
Y_pred = Y_pred.astype(int) 

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
    })
submission["Survived"]=Y_pred
submission.to_csv('output/submission.csv', index=False)





from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),\
                        np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))
plt.contourf(X1, X2, classifier_ann.predict(np.array([X1.ravel(), X2.ravel()] ).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j,1],
                c = ListedColormap(('red','green'))(i),label =j  )
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()






# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Model Logistic Regression
from sklearn.linear_model import LogisticRegression 
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)

coeff_df = pd.DataFrame(train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

print(coeff_df.sort_values(by='Correlation', ascending=False))


# Support Vector Machines
from sklearn.svm import SVC, LinearSVC

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print(acc_svc)

# k-Nearest Neighbors algorithm (or k-NN for short) is
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(acc_knn)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print(acc_gaussian)


# Perceptron
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print(acc_perceptron)

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(acc_linear_svc)

# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print(acc_sgd)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)
# Decision Tree
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(acc_decision_tree)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
classifier.score(X_train, Y_train)
acc_random_forest = round(classifier.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)

#Model evaluation

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))
#submission
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
    })
submission["Survived"]=Y_pred
submission.to_csv('output/submission.csv', index=False)

