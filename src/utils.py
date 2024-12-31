from src.exception import CustomException
from src.logger import logging
import pickle
import os
import sys
from sklearn.metrics import accuracy_score,confusion_matrix

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        logging.info('Error occured in save_object')
        raise CustomException(e,sys)

def eval_model(X_train,y_train,X_test,y_test,models:dict):
    try:
        report = {}
        for i in range(len(models)):
            model  = list(models.values())[i]
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test,y_pred)
            report[list(models.keys())[i]]= (cm[1][1]/(cm[1][1]+cm[1][0]))
        return report
    
    except Exception as e:
        logging.info('Error occured in eval_model')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        logging.info('Error occured in load_object')
        raise CustomException(e,sys)