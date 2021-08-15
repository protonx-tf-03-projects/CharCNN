import numpy as np 
import pandas as pd 
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

class Dataset:
    def __init__(self, test_size):
        self.test_size = test_size 
        self.tokenizer = None 

    def remove_punc(self, text):
        clean_text = re.sub(r'[^\w\s]','', text)
        return clean_text

    def remove_html(self, text):
        cleanr = re.compile('<.*?>')
        clean_text = re.sub(cleanr, '', text)
        return clean_text

    def remove_urls(self, text):
        clean_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        return clean_text

    def preprocess_data(self, text): 
        text = self.remove_punc(text)
        text = self.remove_html(text)
        text = self.remove_urls(text)
        return text

    def build_tokenizer(self, texts, vocab_size):
        tokenizer = Tokenizer(vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        return tokenizer

    def tokenize(self, tokenizer, texts, max_len):
        tensor = tokenizer.texts_to_sequences(texts)
        tensor = pad_sequences(tensor, maxlen=max_len, padding='post')
        return tensor 

    def get_max_len(self, texts):
        return max([len(sentence.split()) for sentence in texts])

    def load_dataset(self, data_path, text_column, label_column):

        datastore = pd.read_csv(
            data_path,
            usecols=[text_column, label_column], 
        )

        dataset = datastore[text_column].tolist()
        label_dataset = datastore[label_column].tolist()

        dataset = [self.preprocess_data(text) for text in dataset]

        return dataset, label_dataset 
    
    def build_dataset(self, data_path, text_column, label_column):
        dataset, label_dataset = self.load_dataset(data_path, text_column, label_column)
        print(dataset[0])
        # split data 
        size = int(len(dataset) * (1 - self.test_size)) 
        self.x_train = dataset[:size]
        self.x_val = dataset[size:]
        self.y_train = label_dataset[:size]
        self.y_val = label_dataset[size:]
        self.vocab_size = len(self.x_train)
        # build tokenizer 
        self.tokenizer = self.build_tokenizer(self.x_train, self.vocab_size)
        # get max_len 
        self.max_len = self.get_max_len(self.x_train)
        # tokenizing 
        self.x_train = self.tokenize(self.tokenizer, self.x_train, self.max_len)
        self.x_val = self.tokenize(self.tokenizer,self.x_val, self.max_len)

        return self.x_train, self.x_val, self.y_train, self.y_val