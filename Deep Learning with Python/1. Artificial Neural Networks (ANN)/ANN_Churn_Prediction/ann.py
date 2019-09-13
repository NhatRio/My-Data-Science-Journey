# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

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
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])# column [2] for gender
onehotencoder = OneHotEncoder(categorical_features = [1])# create dummy variable for countries column
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:] #remove the 1st column to avoid dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part 2: Making the ANN

#Import the Keras libraries and packages
import keras 
from keras.models import Sequential  #initialize the neural network
from keras.layers import Dense # bulid the layers of ANN

#Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11)) 
        # the 1st hidden layer has 6 nodes
        #'uniform': initialize the weights randomly and close to zero
        # 'relu': rectifier activation function
        ####### NOTE: rectifier for hidden layers and sigmoid func for output layer
# Adding the 2nd hidden layer
classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu')) 

        
# Adding the output layer
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid')) 
        ####### NOTE: for the output more 2 categories (ex:3) we have to change the number
                # and the activation func by 3 and 'softmax'(softmax is the sigmoid
                    #function to three or more categories output)  
 #Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
        #optimizer: algorithm to find the optimal set of weights
        #loss: for the output more 2 categories (ex:3) categorical_crossentropy
        #metrics: criterion to evaluate our model
                                       
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
            #bacth_size: the number of observations after  which you want to update the weights
            #epochs : number of rounds that the whole training set pass through the ANN

#Making the predictions and evaluating the model
 
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)