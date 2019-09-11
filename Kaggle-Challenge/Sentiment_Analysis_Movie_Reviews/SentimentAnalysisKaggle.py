# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('dataset/train.tsv', delimiter = '\t', quoting = 3)
#choose the delimiter by 'Tab' and ignore the quotes ' and "

#Cleaning the texts
import re
import nltk
#nltk.download('stopwords') #download the top non-significant words 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #for stemming
corpus = []
for i in range(0,dataset.shape[0]):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i]) #removing all except a-zA-Z letters and replace by space
    review = review.lower() #make all letters to lower case
     
    #remove non-significant words (the, a, an, on, and, in...) + stemming: taking the root of the word
    review = review.split() # split the review to the list of word
    ps = PorterStemmer() 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # only for the englsih word
                                                                            # set() here for running faster because 
    #Joining the word list to a sentence
    review = ' '.join(review) #seperate by  space                                                                          
    corpus.append(review)
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer #tokenization: task of chopping a character sequence
                                                                #into piece, called Tokens
cv = CountVectorizer(max_features = 1500)#max_feature for filtering the non-relevant words   
                                          #just take  the 1500 most frequent words                                    
X =  cv.fit_transform(corpus).toarray() #creating the sparse matrix  with each columne presenting for one word                         
                                       # toarray to make X be a matrix 
y = dataset.iloc[:,1].values   
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)




# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print('Accuracy =', (cm[0,0]+cm[1,1])/np.sum(cm))
print('Precision =', cm[0,0]/np.sum(cm[:,0]))
print('Recall = ',cm[0,0]/np.sum(cm[0,:]))
print('F1_Score =', 2*Precision*Recall/(Precision+Recall))