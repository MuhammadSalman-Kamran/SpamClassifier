import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import data_splitting
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    data_file_path = os.path.join('artifacts', 'data.csv')
    train_data_file_path  = os.path.join('artifacts', 'train.csv')
    test_data_file_path = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def ingestion(self):
        logging.info('Ingestion of the data has started')
        try:
            logging.info('Importing the dataset')
            df = pd.read_csv('notebook/data/Clean/Clean_data.csv')
            logging.info('Making the directory for storing csv files')
            os.makedirs(os.path.dirname(self.data_ingestion_config.data_file_path), exist_ok=True)
            logging.info('Storing the Sample Data')
            df.to_csv(self.data_ingestion_config.data_file_path, index=False, header=True)

            logging.info('Diving the data into training and testing files')
            train_df, test_df = data_splitting(df, 0.2)

            logging.info('Saving Training CSV file')
            train_df.to_csv(self.data_ingestion_config.train_data_file_path, index = False, header = True)
            logging.info('Saving Testing CSV file')
            test_df.to_csv(self.data_ingestion_config.test_data_file_path, index = False, header = True)


            return(
                self.data_ingestion_config.train_data_file_path,
                self.data_ingestion_config.test_data_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)