import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
import joblib
from dataclasses import dataclass
from src.utils import save_object, evaluate_model

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    model_path: str=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "LinearRegression":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
                }

            model_report:dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No Best Model Found")

            save_object(
                file_path=self.model_trainer_config.model_path,
                obj = best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square



        except Exception as e:
            raise CustomException(e,sys)
