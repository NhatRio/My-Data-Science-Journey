# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

########### Part 1 - Building the CNN ##################

#Importing the Keras libraries and packages
from keras.models import Sequential  #initialize the neural network
    # Keras has two kinds of models, Sequential and Graph.
    # Sequential models are appropriate for traditional feedforward NN,
    # in which units from any given layer only feed forward into the next layer. 
    #Graph models are appropriate for more complex NNs, such as recursive NNs.
from keras.layers import Convolution2D # convolution step in which we add convolutional layers
                                                #and the images are in 2D
from keras.layers import MaxPooling2D #pooling step: adding the pooling layers
from keras.layers import Flatten #flattening step: converting all the pool feature maps into
                                    # large feature vector becoming the input of fully connected layers 
from keras.layers import Dense # bulid the layers of ANN: adding the fully connected layers and
                                # classic ANN

#Initialising the CNN
classifier = Sequential()                                
                                
# - Step 1: Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation = 'relu'))
        #filters: nb of feature detector = nb of feature map=32
        #kernel_size = size of feature detector matrix= nb of row, nb of column=3 ,3  
        #input_shape( 256, 256, 3): unique size of our input images want to convert to
                    #3: for color image and 256x256 pixel
                    # tensorflow backend: 64, 64, 3
                    # theano backend: 3, 64, 64    
        # activation function: to make sure we don't have any negative pixel
            #values in our feature map in order to have the non-linearity in CNN               

# -Step 2: Max Pooling : To reduce the nb of nodes for the next step
classifier.add(MaxPooling2D(pool_size = (2, 2)))            

#Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    #we don't need the  input_shape here since we have something previous that are feature maps
classifier.add(MaxPooling2D(pool_size = (2, 2)))            


# -Step 3: Flattening
classifier.add(Flatten())

# -Step 4: Full connection = adding fully connected layer = hidden+output layer
classifier.add(Dense(output_dim = 128, activation = 'relu' )) #hidden layer
        #do not choose too small or too big out_dim, around 100 is a good choice(2^n)
classifier.add(Dense(output_dim = 1, activation = 'sigmoid',  )) #output layer

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy' , mmetrics = ['accuracy'])
#two reasons why we choose 'binary_crossentropy':
    #1) this func corresponds to the Logarithmic Loss that is the lost func we use
        # in general for classification problems like logistic regression.
    #2) we have the binary outcome    
    ####NOTE: for the output more 2 categories (ex:3) categorical_crossentropy
    

######### Part 2: Fitting the CNN to the images #################################
#Image augmentation(encrich our dataset- training set) to prevent overfitting
 #Why we use image augementation: 1 reason of overfitting is having few of data to train
  #we have 8000 images for training set, that is not much, so we rotate them, flip or shift or even 
      #shear them
#link: https://keras.io/preprocessing/image/      
from keras.preprocessing.image import ImageDataGenerator 
train_datagen = ImageDataGenerator(
        rescale=1./255, #make all pixel values in the training images be between 0 and 1 
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) #make all pixel values in the test images be between 0 and 1 

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64), 
                                                #dimensions expected by CNN
                                                batch_size=32,
                                                # the batches in which some random samples of our images included                                                                   
                                                 #that contains the number of images that will go through CNN                                                                        
                                                 #after which, the weights will be updated
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=8000,#the nb of image in training set(all 8000 images pass throught CNN)
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000)#the nb of image in test set(all 8000 images pass throught CNN)


#There are 2 ways to impove the accuracy that are: 
    #1) add another convolutional layer
    #2) add more fully connected layers
    #Morever, we can get bigger target_size because when we augment them, we can get more information