import os
import sys
import string
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import processing, save_obj
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

@dataclass
class DataTransformationConfig:
    processing_file_path = os.path.join('artifacts', 'processing.pkl')
    vectorizer_file_path = os.path.join('artifacts','vectorizer.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
    
    def transformation(self, train_file_path, test_file_path):
        train_df = pd.read_csv(train_file_path)
        test_df = pd.read_csv(test_file_path)

        train_df['transformed_text'] = train_df['transformed_text'].fillna("")
        test_df['transformed_text'] = test_df['transformed_text'].fillna("")

        train_input = train_df['transformed_text']
        x_train = tfidf.fit_transform(train_input).toarray()
        test_input = test_df['transformed_text']
        x_test = tfidf.transform(test_input).toarray()

        save_obj(self.data_transformation_config.processing_file_path, processing)
        save_obj(self.data_transformation_config.vectorizer_file_path, tfidf)

        return (x_train, x_test)
        

# if __name__ == '__main__':
#     tran_obj = DataTransformation()
#     output = tran_obj.processing('Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...')

#     print(output)
