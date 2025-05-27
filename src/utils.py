import os
import sys

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import dill


from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
  
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            logging.info(f"Training {list(models.keys())[i]} model")
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            logging.info(f"{list(models.keys())[i]} model score: {test_model_score}")
            logging.info(f"{list(models.keys())[i]} model train score: {train_model_score}")
            logging.info(f"Best parameters for {list(models.keys())[i]}: {gs.best_params_}")
            logging.info(f"Best score for {list(models.keys())[i]}: {gs.best_score_}")
            logging.info("=============================================================")

        logging.info("Model evaluation completed")
        logging.info(f"Model report: {report}")
        return report

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)