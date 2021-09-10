import os
from argparse import ArgumentParser
from data import Dataset
import pandas as pd 
from model import *
import tensorflow as tf
from constant import *
from tensorflow.keras.optimizers import *
tf.config.experimental_run_functions_eagerly(True)
if __name__ == "__main__":
    parser = ArgumentParser()
    home_dir = os.getcwd()    
    # Arguments users used when running command lines
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--mode",default= "small", type= str)
    parser.add_argument("--vocab-folder", default= '{}/saved_vocab/CharCNN/'.format(home_dir), type= str)
    parser.add_argument("--train-file", default= 'data.csv', type= str)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--embedding-size", default=100, type=int)
    parser.add_argument("--test-size", default=0.3, type=float)
    parser.add_argument("--num-classes", default=2, type=float)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--smallCharCNN-folder", default="smallCharCNN", type=str)
    parser.add_argument("--largeCharCNN-folder", default="largeCharCNN", type=str)
    parser.add_argument("--padding", default="same", type=str)

    args = parser.parse_args()

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


    dataset = Dataset(vocab_folder= args.vocab_folder)
    x_train, x_val, y_train, y_val = dataset.build_dataset(data_path = args.train_file, test_size= args.test_size)

    # Initializing models
    # Small-CharCNN
    small_CharCNN = CharCNN(dataset.vocab_size, args.embedding_size, dataset.max_len, args.num_classes, feature = "small", padding= args.padding)
    # Large-CharCNN
    large_CharCNN = CharCNN(dataset.vocab_size, args.embedding_size, dataset.max_len, args.num_classes, feature = "large", padding= args.padding)

    # Set up loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # Optimizer Definition
    adam = tf.keras.optimizers.Adam(learning_rate= args.learning_rate)

    # Compile optimizer and loss function into models
    small_CharCNN.compile(optimizer= adam, loss = loss, metrics  = [metric])
    large_CharCNN.compile(optimizer= adam, loss = loss, metrics = [metric])

    # Do Training model
    if args.mode == 'small' or args.mode == 'all':
        print("-------------Training Small CharCNN------------")
        small_CharCNN.fit(x_train,y_train,validation_data= (x_val,y_val), epochs= args.epochs, batch_size= args.batch_size, validation_batch_size= args.batch_size)
        print("----------Finish Training Small CharCNN--------")
        # Saving models
        small_CharCNN.save(args.smallCharCNN_folder)
    if args.mode == 'large' or args.mode == 'all':
        print("-------------Training Large CharCNN------------")
        large_CharCNN.fit(x_train,y_train, validation_data= (x_val,y_val), epochs= args.epochs, batch_size = args.batch_size, validation_batch_size = args.batch_size)
        print("----------Finish Training Large CharCNN--------")
        # Saving models
        large_CharCNN.save(args.largeCharCNN_folder)
