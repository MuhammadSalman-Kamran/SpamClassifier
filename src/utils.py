import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import string

def data_splitting(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return (train_df, test_df)

def save_obj(preprocess_file_path, obj):
    try:
        logging.info('Making Directory for storing the object')
        os.makedirs(os.path.dirname(preprocess_file_path), exist_ok=True)
        logging.info('Saving the object ')
        pickle.dump(obj, open(preprocess_file_path, 'wb'))
    except Exception as e:
        raise CustomException(e, sys)
    
def load_obj(file_path):
    try:
        logging.info('Loading the object')
        return pickle.load(open(file_path, 'rb'))
    
    except Exception as e:
        raise CustomException(e, sys)
    
def processing(text):
        text = text.lower()
        text = nltk.word_tokenize(text)
        
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)
        text = y[:]
        y.clear()
        
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
                
        text = y[:]
        y.clear()
        
        for i in text:
            y.append(ps.stem(i))
                
        return " ".join(y)