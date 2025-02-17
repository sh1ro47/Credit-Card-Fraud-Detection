import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestionConfig:
    def __init__(self):
        self.train_data_path = os.path.join('artifacts','train.csv')
        self.test_data_path = os.path.join('artifacts','test.csv')
        self.raw_data_path = os.path.join('artifacts','raw.csv')
        

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info('Data ingestion begins')
        try:
            df = pd.read_csv('Notebook\data\creditCardFraud_28011964_120214.csv')
            logging.info('Dataset read successfully')
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            
            train_set, test_set = train_test_split(df,test_size=0.30,random_state=42)
            logging.info('Train test split done successfully')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info('Data ingestion completed')
            
            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
        
        except Exception as e:
            logging.info('Error occured in initiate_data_ingestion')
            raise CustomException(e,sys)