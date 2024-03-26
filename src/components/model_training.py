import sys
import os
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_obj
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score

@dataclass
class ModelTrainingConfig:
    model_file_path = os.path.join('artifacts','model.pkl')

class ModelTraining:
    def __init__(self) -> None:
        self.model_training_config = ModelTrainingConfig()

    def training(self, input_train_arr, input_test_arr,train_file_path, test_file_path):
        logging.info('Training Process has started')
        try:
            logging.info('Training and Testing file imported successfully')
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)

            logging.info('Converting output into numpy array')
            train_output = train_df['target'].values
            test_output = test_df['target'].values

            model = MultinomialNB()
            model.fit(input_train_arr, train_output)
            model_pred = model.predict(input_test_arr)

            precision = precision_score(test_output, model_pred)
            print(precision)
            accuracy = accuracy_score(test_output, model_pred)
            print(accuracy)

            logging.info('Dumping the model obj')
            save_obj(self.model_training_config.model_file_path, model)

            logging.info('Returning the model object folder path')
            return self.model_training_config.model_file_path
        
        except Exception as e:
            raise CustomException(e, sys)