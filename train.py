import os
from argparse import ArgumentParser
from data import Dataset
import pandas as pd 

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)

    home_dir = os.getcwd()
    args = parser.parse_args()

    # FIXME
    # Project Description

    print('---------------------Welcome to ${name}-------------------')
    print('Github: ${accout}')
    print('Email: ${email}')
    print('---------------------------------------------------------------------')
    print('Training ${name} model with hyper-params:') # FIXME
    print('===========================')

    # Load data 
    data_path = "/content/IMDB Dataset.csv"
    text_column = "review"
    label_column = "sentiment"
    imdbd_dataset = Dataset(test_size=0.2)
    x_train, x_val, y_train, y_val = imdbd_dataset.build_dataset(data_path, text_column, label_column)
    print(x_train)
    
    # FIXME
    # Do Prediction


