import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

import joblib
from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataTransformationConfig:
    X_train_data_path: str=os.path.join('artifacts',"x_train.csv")
    X_test_data_path: str=os.path.join('artifacts',"x_test.csv")
    y_train_data_path: str=os.path.join('artifacts',"y_train.csv")
    y_test_data_path: str=os.path.join('artifacts',"y_test.csv")

class DataTransformation:
    def __init__(self):
        self.transformation_config =DataTransformationConfig()

    def initiate_data_transformation(self, train_data,test_data):
        logging.info("Entered the data tranformation component")
        try:
            train = pd.read_csv(train_data) #pd.read_csv("artifacts/train.csv")
            test = pd.read_csv(test_data) #pd.read_csv("artifacts/test.csv")
            logging.info("Train and test data loaded")

            os.makedirs(os.path.dirname(self.transformation_config.X_train_data_path),exist_ok=True)

            logging.info("Setting Initial X and y train/test variables")
            X_train = train.drop("math_score",axis=1)
            y_train = train['math_score']
            y_train.to_csv(self.transformation_config.y_train_data_path,index=False,header=True)

            X_test = test.drop("math_score",axis=1)
            y_test = test['math_score']
            y_test.to_csv(self.transformation_config.y_test_data_path,index=False,header=True)


            ## Column Transformers:
            num_columns = [col for col in X_train.columns if X_train[col].dtypes != 'O']
            cat_columns = [col for col in X_train.columns if X_train[col].dtypes == 'O']

            num_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()

            preprocessor = ColumnTransformer(
                [
                    ('OneHotEncoder',oh_transformer,cat_columns),
                    ('StandardScaler',num_transformer,num_columns)
                ]
            )

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            # Convert to DataFrame and save
            pd.DataFrame(X_train).to_csv(self.transformation_config.X_train_data_path, index=False, header=True)
            pd.DataFrame(X_test).to_csv(self.transformation_config.X_test_data_path, index=False, header=True)

            joblib.dump(preprocessor, 'artifacts/preprocessor.pkl')
            logging.info("Data transformation completed successfully")

            return X_train,X_test,preprocessor

        except Exception as e:
            raise CustomException(e,sys)



