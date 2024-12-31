import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import eval_model

class ModelTrainerConfig:
    def __init__(self):
        self.model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Initiating model training')
            X_train,y_train,X_test,y_test = (train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])
            
            models = {
                'LogisticRegression':LogisticRegression(),
                'GaussianNB':GaussianNB(),
                'RandomForestClassifier':RandomForestClassifier(),
                'KNeighborsClassifier':KNeighborsClassifier(),
                'XGBClassifier':XGBClassifier(),
                'SVC':SVC()
            }
            
            model_report = eval_model(X_train,y_train,X_test,y_test,models)
            
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            
            best_model_score = max(sorted(model_report.values()))
            best_model_index = list(model_report.values()).index(best_model_score)
            best_model_name = list(model_report.keys())[best_model_index]
            best_model = models[best_model_name]
            
            print(f'Best Model Found , Model Name : {best_model_name} , Precision of class 1 : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Precision of class 1 : {best_model_score}')
            
            save_object(self.model_trainer_config.model_path,best_model)
            logging.info('Model saved')
        
        except Exception as e:
            logging.info('Error occured in initiate_model_trainer')
            raise CustomException(e,sys)