import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


from src.exception import CustomException
from src.logger import logging
from src.utils import save_model
from src.utils import evalute_model
import os,sys
from dataclasses import dataclass
from  src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file = os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting Dependant and Independent variables from train and test data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            print("start model training")

            models = {
                'RF' : RandomForestClassifier(),
                'DTC':DecisionTreeClassifier(),
                'logistic':LogisticRegression()
            }
        
            model_report:dict = evalute_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print("\n============================================================================\n")
            logging.info(f'Model report : {model_report}')

        #To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)

            ]
            best_model = models[best_model_name]
            print(f'Best model found,Model name: {best_model_name},Accuracy_score: {best_model_score}')
            print('\n=========================================================\n')
            logging.info(f'Best model found,Model name: {best_model_name},Accuracy_score: {best_model_score}')


            save_model(
                file_path = self.model_trainer_config.trained_model_file,
                obj = best_model
            )
        
        except Exception as e:
            logging.info('Exception occured at model Training')
            raise CustomException(e,sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer_obj = ModelTrainer()
    model_trainer_obj.initiate_model_training(train_arr,test_arr)

