import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.ops.gen_math_ops import mod 

class CharCNN(tf.keras.models.Model):
    def __init__(self,vocab_szie = 10000, embedding_size = 100, max_length = 2000, num_classes = 2,feature = 'small'):
        super(CharCNN,self).__init__()
        self.num_classes = num_classes

        assert feature in ['small','large']
        if feature == 'small':
            self.units_fc = 1024
            self.num_filter = 256
            self.stddev = 0.05
        else:
            self.units_fc = 2048
            self.num_filter = 1024
            self.stddev = 0.02
        self.vocab_size = vocab_szie
        self.embedding_size = embedding_size
        self.max_length = max_length
    def call(self, data):
        x = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size)(data) # embedding layer
        # block Convolutional layers
        #layer 1
        x = Conv1D(filters= self.num_filter, kernel_size= 7,kernel_initializer= tf.keras.initializers.RandomNormal(mean = 0., stddev= self.stddev, seed= 2021), activation="relu")(x)
        x = MaxPooling1D(pool_size= 3)(x)
        #layer 2
        x = Conv1D(filters= self.num_filter, kernel_size= 7,kernel_initializer= tf.keras.initializers.RandomNormal(mean = 0., stddev= self.stddev, seed= 2021), activation="relu")(x)
        x = MaxPooling1D(pool_size= 3)(x)
        #layer 3
        x = Conv1D(filters= self.num_filter, kernel_size= 3,kernel_initializer= tf.keras.initializers.RandomNormal(mean = 0., stddev= self.stddev, seed= 2021), activation="relu")(x)
        #layer 4
        x = Conv1D(filters= self.num_filter, kernel_size= 3,kernel_initializer= tf.keras.initializers.RandomNormal(mean = 0., stddev= self.stddev, seed= 2021), activation="relu")(x)
        #layer 5 
        x = Conv1D(filters= self.num_filter, kernel_size= 3,kernel_initializer= tf.keras.initializers.RandomNormal(mean = 0., stddev= self.stddev, seed= 2021), activation="relu")(x)
        #layer 6 
        x = Conv1D(filters= self.num_filter, kernel_size= 3,kernel_initializer= tf.keras.initializers.RandomNormal(mean = 0., stddev= self.stddev, seed= 2021), activation="relu")(x)
        x = MaxPooling1D(pool_size= 3)(x)
        # end Convolutional layers
        x = Flatten()(x) # flatten layer
        # block Full-Connected layers
        x = Dense(self.units_fc,activation= 'relu')(x)
        x = Dropout(0.5)(x) # Droppout layer 1
        x = Dense(self.units_fc,activation= 'relu')(x)
        x = Dropout(0.5)(x)  # Droppout layer 2
        x = Dense(units= self.num_classes, activation= 'softmax')(x)
        return x
        # end 
      
