import pandas as pd
import os
import sys
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.utils import save_object


class Models_report:
    def print_report(self):
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()


        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_path,test_path)

        trainer = ModelTrainer()
        r2, report = trainer.initiate_model_trainer(train_arr,test_arr)
        
        save_object(
                file_path=os.path.join("artifacts","score.pkl"),
                obj = report
            )
        # return report

        


if __name__=="__main__":
    report = Models_report().print_report()
    print(report)




