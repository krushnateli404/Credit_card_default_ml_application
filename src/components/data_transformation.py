import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass
from src.utils import save_model
import sys

@dataclass
class DataTransformationConfig:
    preprocesser_obj_file = os.path.join("artifact","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformation_object(self):
        try:
            logging.info("Data Transforamtion initiated")

            numerical_column = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
            'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
            'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
            'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            
            logging.info('Pipeline Initiated')
            
            num_pipeline = Pipeline(
            steps=[
                   ('imputer',SimpleImputer(strategy='median')),
                   ('scalar',StandardScaler())
                ]
                )
            
            Preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_column)
                    ]
                    )
            return Preprocessor
            logging.info("pipline completed")
        
        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
    

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info(f"Train dataframe head: \n{train_df.head().to_string()}" )
            logging.info(f"Test dataframe head: \n{test_df.head().to_string()} ")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformation_object()
            target_column_name = 'default.payment.next.month'
            drop_columns = [target_column_name,"ID"]
            input_features_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]
            ##Transforming using preprocessor obj
            print(input_features_train_df)
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Applying preprocessing object on training and testing datasets.")
            train_arr = np.c_[input_features_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            save_model(
                file_path = self.data_transformation_config.preprocesser_obj_file,
                obj = preprocessing_obj
                )
            logging.info("Preprocessor pickle file saved")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesser_obj_file
                )
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

