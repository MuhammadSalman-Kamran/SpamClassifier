import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_obj

class Prediction:
    def __init__(self) -> None:
        pass

    def prediction(self, input):
        logging.info('Into the Prediction Function')
        try:
            logging.info('Defining the paths of model, vectorizer and preprocessing obj')
            model_file_path = os.path.join('artifacts', 'model.pkl')
            vector_file_path = os.path.join('artifacts', 'vectorizer.pkl')
            process_file_path = os.path.join('artifacts', 'processing.pkl')

            logging.info('Loading the Processing object')
            processor = load_obj(process_file_path)
            logging.info('Processig the vectorizer object')
            vectorizer = load_obj(vector_file_path)
            logging.info('Loading the model for prediction')
            model = load_obj(model_file_path)

            logging.info('Preprocessing the user input')
            processed_input = processor(input)
            logging.info('Converting the text into vector')
            vectorize_input = vectorizer.transform([processed_input])
            logging.info('Making the prediction on the User input')
            prediction = model.predict(vectorize_input)[0]

            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)