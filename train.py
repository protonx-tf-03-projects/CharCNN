import os
from argparse import ArgumentParser
from data import Dataset
import pandas as pd 
from model import *
from tensorflow.keras.optimizers import *
import urllib.request
tf.config.experimental_run_functions_eagerly(True)
if __name__ == "__main__":
    parser = ArgumentParser()
    
    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--embedding-size", default=100, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--num-classes", default=2, type=float)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--smallCharCNN-folder", default="smallCharCNN", type=str)
    parser.add_argument("--largeCharCNN-folder", default="largeCharCNN", type=str)


    home_dir = os.getcwd()
    args = parser.parse_args()

    # FIXME
    # Project Description

    print('---------------------Welcome to CharCNN-------------------')
    print("Team Leader")
    print("1. Github: hoangcaobao")
    print("Team member")
    print('1. Github: Nguyendat-bit')
    print('2. Github: aestheteeism')
    print('---------------------------------------------------------------------')
    print('Training CharCNN model with hyper-params:') 
    for i, arg in enumerate(vars(args)):
        print('{}. {}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')

    # Load data 
    print("-------------TRAINING DATA------------")
   
    data_path = "IMDB Dataset.csv"
    text_column = "review"
    label_column = "sentiment"
    imdbd_dataset = Dataset(test_size=args.test_size)
    x_train, x_val, y_train, y_val = imdbd_dataset.build_dataset(data_path, text_column, label_column)
    

   