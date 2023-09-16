import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from src.logger import logging

class DataIngestionConfig():
    train_data_path = os.path.join('artifact','train.csv')
    test_data_path = os.path.join('artifact','test.csv')
    raw_data_path = os.path.join('artifact','raw.csv')


class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    
    def initiate_data_ingestion(self):
        logging.info("Start data ingestion")        
        try:
            df = pd.read_csv('https://raw.githubusercontent.com/Larissavvy/Credit-Card-Default-Prediction/main/UCI_Credit_Card.csv')
            logging.info("Raw data reading completely...")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path)

            
            train_data,test_data = train_test_split(df,test_size=0.25,random_state=40)
            logging.info("train and test spit done...")

            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data ingestion complete....")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )




        
        except Exception as e:
            raise Exception

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

