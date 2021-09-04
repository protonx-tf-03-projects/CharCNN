
from tensorflow.keras.layers import *
import tensorflow as tf

class CharCNN(tf.keras.models.Model):
    def __init__(self,vocab_size,embedding_size,max_length,num_classes,feature = 'small',padding = 'same'):
        super(CharCNN,self).__init__()
        assert feature in ['small','large']
        assert padding in ['valid','same']
        self.padding = padding 
        self.num_classes = num_classes

        if feature == 'small':
            self.units_fc = 1024 #  The number of output units of fully-connected layer
            self.num_filter = 256
            self.stddev = 0.05 # standard deviation
        else:
            self.units_fc = 2048 # The number of output units of fully-connected layer
            self.num_filter = 1024
            self.stddev = 0.02 # standard deviation

        # initialize the weights using a Gaussian distribution for conv layer
        # The mean and standard deviation used for initializing the large model is (0, 0.02) and small model (0, 0.05).
        self.initializers = tf.keras.initializers.RandomNormal(mean = 0., stddev= self.stddev, seed = 42)

        # defind vocab_size,embedding_size,max_length
        self.vocab_size, self.embedding_size, self.max_length = vocab_size,embedding_size,max_length

        self.embedding = Embedding(self.vocab_size, self.embedding_size,input_length= self.max_length)
        # block Convolutional layers
        # block conv layer 1
        self.conv1d_1 = Conv1D(self.num_filter, kernel_size= 7, kernel_initializer= self.initializers, activation= 'relu',padding = self.padding)
        self.maxpooling1d_1 = MaxPooling1D(3)
        # block conv layer 2
        self.conv1d_2 = Conv1D(self.num_filter, kernel_size= 7, kernel_initializer= self.initializers, activation= 'relu',padding = self.padding)
        self.maxpooling1d_2 = MaxPooling1D(3)
        # block conv layer 3
        self.conv1d_3 = Conv1D(self.num_filter, kernel_size= 3, kernel_initializer= self.initializers, activation= 'relu',padding = self.padding)
        # block conv layer 4
        self.conv1d_4 = Conv1D(self.num_filter, kernel_size= 3, kernel_initializer= self.initializers, activation= 'relu',padding = self.padding)
        # block conv layer 5
        self.conv1d_5 = Conv1D(self.num_filter, kernel_size= 3, kernel_initializer= self.initializers, activation= 'relu',padding = self.padding)
        # block conv layer 6
        self.conv1d_6 = Conv1D(self.num_filter, kernel_size= 3, kernel_initializer= self.initializers, activation= 'relu',padding = self.padding)
        self.maxpooling1d_6 = MaxPooling1D(3)
        
        self.flatten = Flatten() # flatten layer

        # block Fully-connected layers
        self.fc1 = Dense(self.units_fc,activation= 'relu')
        self.drp1 = Dropout(0.5)
        self.fc2 = Dense(self.units_fc,activation= 'relu')
        self.drp2 = Dropout(0.5)
        self.fc3 = Dense(self.num_classes, activation= 'softmax')
        
    def call(self,data):
        x = self.embedding(data)
        x = self.maxpooling1d_1(self.conv1d_1(x))
        x = self.maxpooling1d_2(self.conv1d_2(x))
        x = self.conv1d_3(x)
        x = self.conv1d_4(x)
        x = self.conv1d_5(x)
        x = self.maxpooling1d_6(self.conv1d_6(x))

        x = self.flatten(x)
         # linear layer 
        x = self.drp1(self.fc1(x))
        # linear layer 
        x = self.drp2(self.fc2(x))
        # ouput layer
        x = self.fc3(x)
        return x
