from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=='__main__':
    data_ingestion = DataIngestion()
    train_path,test_path = data_ingestion.initiate_data_ingestion()
    print(train_path,test_path)
    data_transformation = DataTransformation()
    preprocessor_obj_path,train_arr,test_arr = data_transformation.initiate_data_transformation(train_path,test_path)
    print(preprocessor_obj_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr,test_arr)