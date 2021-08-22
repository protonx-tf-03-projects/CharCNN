from ast import parse
import os
from argparse import ArgumentParser
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras.layers import embeddings 
from data import *
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from constant import *
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--smallCharCNN-folder", default="smallCharCNN", type=str)
    parser.add_argument("--largeCharCNN-folder", default="largeCharCNN", type=str)
    parser.add_argument("--test-file", default="test.csv", type=str)
    parser.add_argument("--model", default="small", type=str)
    parser.add_argument("--result-file", default="result.csv", type=str)
    parser.add_argument("--test-text-column", default="sentences", type=str)
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

    # Loading Models 
    # Small-CharCNN model
    if(args.model=="small"):
        model = tf.keras.models.load_model(args.smallCharCNN_folder)
    # Large-CharCNN model
    else:
        model = tf.keras.models.load_model(args.largeCharCNN_folder)

    # Load train data to have some importants parameters
    data_path = data_path
    text_column = text_column
    label_column = label_column
    imdbd_dataset = Dataset(test_size=args.test_size)
    x_train, x_val, y_train, y_val = imdbd_dataset.build_dataset(data_path, text_column, label_column)

    # Load test sentences 
    sentence = pd.read_csv(args.test_file)
    sentence = np.array(sentence[args.test_text_column])
    test = [imdbd_dataset.preprocess_data(i) for i in sentence]
    test = imdbd_dataset.tokenizer.texts_to_sequences(test)
    test = pad_sequences(test, maxlen=imdbd_dataset.max_len, padding=padding)

    # Preidct
    print("================Predicting================")
    predict=model.predict(test)
    predict=np.argmax(predict, axis=1)

    #Decode predict
    result=[]
    for i in predict:
        if(i==0):
            result.append("negtive")
        else:
            result.append("positive")

    #Save to csv
    data=[]
    for i in range(0, len(result)):
        data.append([sentence[0], result[i]])
    df=pd.DataFrame(data, columns=[args.test_text_column, "sentiment"])
    df.to_csv(args.result_file, index=False)
    print("End Predicting. Now your result will be in {}".format(args.result_file))

