import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
import joblib
from dataclasses import dataclass
from src.utils import save_object

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline



@dataclass
class DataTransformationConfig:
    preprocessor_path: str=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config =DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation object.

        '''

        logging.info("Entered the data tranformation component")
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("Scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps= [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("OneHot",OneHotEncoder()),
                    ("Scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Encoding Step completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_transformer",num_pipeline,numerical_columns),
                    ("cat_transformer",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            logging.info("Train and test data loaded")

            preprocessor = self.get_data_transformer_object()

            target_column_name = "math_score"
            
            X_train = train.drop(columns = [target_column_name],axis=1)
            y_train = train[target_column_name]

            X_test = test.drop(columns = [target_column_name],axis=1)
            y_test = test[target_column_name]

            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_processed,np.array(y_train)]
            test_arr = np.c_[X_test_processed,np.array(y_test)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_path,
                obj = preprocessor
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_path)

        except Exception as e:
            raise CustomException(e,sys)


































#########################################
    # def initiate_data_transformation2(self, train_data,test_data):
    #     try:
    #         train = pd.read_csv(train_data)
    #         test = pd.read_csv(test_data)
    #         logging.info("Train and test data loaded")

    #         os.makedirs(os.path.dirname(self.transformation_config.X_train_data_path),exist_ok=True)

    #         logging.info("Setting Initial X and y train/test variables")
    #         X_train = train.drop("math_score",axis=1)
    #         y_train = train['math_score']
    #         y_train.to_csv(self.transformation_config.y_train_data_path,index=False,header=True)

    #         X_test = test.drop("math_score",axis=1)
    #         y_test = test['math_score']
    #         y_test.to_csv(self.transformation_config.y_test_data_path,index=False,header=True)


    #         ## Column Transformers:
    #         num_columns = [col for col in X_train.columns if X_train[col].dtypes != 'O']
    #         cat_columns = [col for col in X_train.columns if X_train[col].dtypes == 'O']

    #         num_transformer = StandardScaler()
    #         oh_transformer = OneHotEncoder()

    #         preprocessor = ColumnTransformer(
    #             [
    #                 ('OneHotEncoder',oh_transformer,cat_columns),
    #                 ('StandardScaler',num_transformer,num_columns)
    #             ]
    #         )

    #         X_train = preprocessor.fit_transform(X_train)
    #         X_test = preprocessor.transform(X_test)

    #         # Convert to DataFrame and save
    #         pd.DataFrame(X_train).to_csv(self.transformation_config.X_train_data_path, index=False, header=True)
    #         pd.DataFrame(X_test).to_csv(self.transformation_config.X_test_data_path, index=False, header=True)

    #         joblib.dump(preprocessor, self.transformation_config.preprocessor_path)
    #         logging.info("Data transformation completed successfully")

    #         return X_train,X_test,preprocessor

    #     except Exception as e:
    #         raise CustomException(e,sys)



