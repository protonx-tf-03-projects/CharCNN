import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import * 



class CharCNN():
    def __init__(self,vocab_szie = 10000, embedding_dim = 64, max_lenght = 2000, num_classes = 10,feature = 'small',padding = 'same'):
        super(CharCNN,self).__init__()
        self.num_classes = num_classes
        assert padding in ['same','valid']
        assert feature in ['small','large']
        if feature == 'small':
            self.units_fc = 1024
            self.num_filter = 256
        else:
            self.units_fc = 2048
            self.num_filter = 1024
        
        self.embedding = Embedding(input_dim=vocab_szie, output_dim= embedding_dim, input_length= max_lenght )
        self.conv1d7 = Conv1D(filters= self.num_filter, kernel_size= 7, padding= padding)
        self.conv1d3 = Conv1D(filters= self.num_filter, kernel_size= 3, padding= padding)
        self.maxpool1d = MaxPooling1D(pool_size= 3, strides= 1)
        self.fc = Dense(self.units_fc,activation= 'relu')
    def call(self, data):
        x = self.embedding(data) # embedding layer
        # block Convolutional layers
        #layer 1
        x = self.conv1d7(x)
        x = self.maxpool1d(x)
        #layer 2
        x = self.conv1d7(x)
        x = self.maxpool1d(x)
        #layer 3
        x = self.conv1d3(x)
        #layer 4
        x = self.conv1d3(x)
        #layer 5 
        x = self.conv1d3(x)
        #layer 6 
        x = self.conv1d3(x)
        x = self.maxpool1d(x)
        # end Convolutional layers
        x = Flatten()(x) # flatten layer
        # block Full-Connected layers
        x = self.fc(x)
        x = Dropout(0.5)(x) # Droppout layer 1
        x = self.fc(x)
        x = Dropout(0.5)(x)  # Droppout layer 2
        x = Dense(units= self.num_classes, activation= 'softmax')
        return x
        # end 

