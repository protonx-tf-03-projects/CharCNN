import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import Embedding,Dense,MaxPooling1D,Conv1D,Dropout
from tensorflow.python.autograph.core.converter import Feature
from tensorflow.python.keras.backend import switch
from tensorflow.python.keras.backend_config import set_floatx

class CharCNN(tf.keras.Model):
    def __init__(self,feature = 'small', num_classes = None,batch_size = None):
        super(CharCNN,self).__init__()
        self.bach_size = batch_size
        self.num_classes = num_classes
        self.kernel_size = [7,7,3,3,3]
        assert feature in ['small','large']
        if feature == 'small':
            self.num_filters = 256
            # the weights using a Gaussian distribution
            self.weight_stddev_initialization = (0,0.02)  
            self.units_fully_connected = 2048
        else:
            self.num_filters = 1024
            self.weight_stdev_initialization =  (0,0.05)
            self.units_fully_connected = 1024
    def convblock():
        pass 
    def fully_connected():
        pass
