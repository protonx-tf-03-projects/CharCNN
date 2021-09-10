import numpy as np 
import pandas as pd 
import re
import io
import os 
import json
from keras.preprocessing.text import tokenizer_from_json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from constant import *

class Dataset:
    def __init__(self, vocab_folder):
        self.vocab_folder = vocab_folder
        self.save_tokenizer_path = '{}tokenizer.json'.format(self.vocab_folder)
        self.save_label_path = 'label.json'
        self.label_dict = None 
        self.tokenizer = None 

        if os.path.isfile(self.save_tokenizer_path):
            # Loading tokenizer
            with open(self.save_tokenizer_path) as f:
                data = json.load(f)
                self.tokenizer = tokenizer_from_json(data)
        
        if os.path.isfile(self.save_label_path):
            # Loading label_dict
            with open(self.save_label_path) as f:
                self.label_dict = json.load(f)

    def remove_punc(self, text):
        #Remove punction in a texts
        clean_text = re.sub(r'[^\w\s]','', text)
        return clean_text

    def remove_html(self, text):
        #Remove html tag in texts
        cleanr = re.compile('<.*?>')
        clean_text = re.sub(cleanr, '', text)
        return clean_text

    def remove_urls(self, text):
        #Remove url link in texts
        clean_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        return clean_text
    def remove_emoji(self, data):
        #Each emoji icon has their unique code
        #Gather all emoji icon code and remove it in texts
        cleanr= re.compile("["
                           u"\U0001F600-\U0001F64F"  
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"  
                           u"\U0001F1E0-\U0001F1FF"  
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        clean_text=re.sub(cleanr, '',data)
        return clean_text
    
    def preprocess_data(self, text): 
        #Use all regex function we create above to clean data
        processors = [self.remove_punc, self.remove_html, self.remove_urls, self.remove_emoji]
        for process in processors:
            text = process(text)
        return text
        
    def build_tokenizer(self, texts, vocab_size):
        #Create tokenizer fit on texts
        tokenizer = Tokenizer(vocab_size, oov_token=oov_token)
        tokenizer.fit_on_texts(texts)
        return tokenizer

    def tokenize(self, tokenizer, texts, max_len):
        #Change all texts to sequence and pad them into 1 size in order to train
        tensor = tokenizer.texts_to_sequences(texts)
        tensor = pad_sequences(tensor, maxlen=max_len, padding=padding)
        return tensor 

    def get_max_len(self, texts):
        #Get maxlen of texts so we can pad sentences without losing information
        return max([len(sentence.split()) for sentence in texts])

    def load_dataset(self, data_path):
        #Get data and label from file csv using pandas
        datastore = pd.read_csv(data_path)
        #Rename column names
        datastore.columns = ['sentence','label']

        dataset = datastore['sentence'].tolist()
        
        #switch to number label
        self.label_dict = dict((l,i) for i,l in enumerate(set(datastore.label.values)))
        label_dataset = datastore['label'].apply(lambda x: self.label_dict[x]).tolist()
        dataset = [self.preprocess_data(text) for text in dataset]
        return dataset, label_dataset 
    
    def build_dataset(self, data_path, test_size):
        """
            This function help to build data that we need to train for CharCNN
            Input:
                data_path: csv file path contain sentences and labels
                text_column: name of column sentences (to extract it from csv)
                label_column: name of column labels (to extract it from csv)
            Output:
                x_train, x_val, y_train, y_val
                All data has been cleaned using regex, token, and pad to same length
        """
        dataset, label_dataset = self.load_dataset(data_path)
        
        # shuffle 
        dataset, label_dataset = shuffle(dataset, label_dataset, random_state = 2111)
      
        # split data 
        size = int(len(dataset) * (1 - test_size)) 
        self.x_train = dataset[:size]
        self.x_val = dataset[size:]
        self.y_train = np.array(label_dataset[:size])
        self.y_val = np.array(label_dataset[size:])
        self.vocab_size = len(self.x_train)
        
        # build tokenizer 
        self.tokenizer = self.build_tokenizer(self.x_train, self.vocab_size)

        # Saving Tokenizer
        print('=============Saving Tokenizer================')
        print('Begin...')
        if not os.path.exists(self.vocab_folder):
            try:
                os.makedirs(self.vocab_folder)
            except OSError as e:
                raise IOError("Failed to create folders")

        tokenizer_json = self.tokenizer.to_json()
        with io.open(self.save_tokenizer_path, 'w', encoding= 'utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii= False))
        print('Done!!!')

        # Saving label dict
        with open('label.json', 'w') as f:
            json.dump(self.label_dict, f)

        # get max_len 
        self.max_len = self.get_max_len(self.x_train)
        
        # tokenizing 
        self.x_train = np.array(self.tokenize(self.tokenizer, self.x_train, self.max_len))
        self.x_val = np.array(self.tokenize(self.tokenizer,self.x_val, self.max_len))
        return self.x_train, self.x_val, self.y_train, self.y_val
