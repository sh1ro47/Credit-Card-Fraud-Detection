import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.utils import save_object

class DataTransformationConfig:
    
    def __init__(self):
        self.preprocessor_obj_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformation_obj(self):
        try:
            logging.info('Initiated data transformation obj process')
            df = pd.read_csv('Notebook\data\creditCardFraud_28011964_120214.csv')
            num_cols = list(df.columns)
            num_pipelines = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            preprocessor_obj = ColumnTransformer([
                ('num_pipeline',num_pipelines,num_cols[:-1])
            ])
            logging.info('Pipeline creation completed')
            
            return preprocessor_obj
        
        except Exception as e:
            logging.info('Error occured in get_data_transformation_obj')
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_data_path,test_data_path):
        
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            
            logging.info('Train and test data read')
            logging.info(f'Train DataFrame Head: \n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head: \n{test_df.head().to_string()}')
            
            preprocessor_obj = self.get_data_transformation_obj()
            logging.info('Obtained preprocessing obj')
            
            target_col = 'default payment next month'
            drop_cols = [target_col]
            
            X_train = train_df.drop(columns=drop_cols,axis=1)
            y_train = train_df[target_col]
            X_test = test_df.drop(columns=drop_cols,axis=1)
            y_test = test_df[target_col]
            
            X_train_arr = preprocessor_obj.fit_transform(X_train)
            X_test_arr = preprocessor_obj.transform(X_test)
            
            train_arr = np.c_[X_train_arr,np.array(y_train)]
            test_arr = np.c_[X_test_arr,np.array(y_test)]
            logging.info('Applied preprocessing object on training and testing datasets.')
            
            save_object(self.data_transformation_config.preprocessor_obj_path,preprocessor_obj)
            logging.info('Preprocessor obj saved')
            
            return self.data_transformation_config.preprocessor_obj_path,train_arr,test_arr
        
        except Exception as e:
            logging.info('Error occured in initiate_data_transformation')
            raise CustomException(e,sys)