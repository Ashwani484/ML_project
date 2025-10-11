import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml,load_data
from scipy.stats import randint
from src.ModelTuner_Classification import ModelTuner
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = get_logger(__name__)
#model training class
class ModelTraining:

    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        self.config = read_yaml(CONFIG_PATH)
        

    #loading and splitting the data
    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")

            # Load train and test DataFrames
            train_df = joblib.load(os.path.join(self.train_path, "train_df.pkl"))
            test_df = joblib.load(os.path.join(self.test_path, "test_df.pkl"))

            # Combine into a single DataFrame
            df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

            # Identify target column from config
            target_column = self.config.get("output_feature")
            if not target_column or target_column not in df.columns:
                raise CustomException(f"Target column '{target_column}' not found in dataset")

            # Split into features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2,  random_state=42
            )

            scaler=StandardScaler()
            X_train=scaler.fit_transform(X_train)
            X_test=scaler.transform(X_test)

            logger.info("Data combined and split successfully for model training")

            return X_train, X_test, y_train, y_test, scaler

        except Exception as e:
            logger.error(f"Error while loading and splitting data: {e}", exc_info=True)
            raise CustomException("Failed to load and split data", e)
            
    #training the model using LightGBM    
    
    #saving the trained model    
    def save_model(self,model,model_name,scaler):
        try:
            #os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)

            logger.info("saving the model")
            
            joblib.dump(model , f"artifacts/models"+f"/{model_name}.pkl")
            
            joblib.dump(scaler , f"artifacts/processed"+f"/scaler.pkl")
            
            logger.info(f"Model saved to {self.model_output_path}")

        except Exception as e:
            logger.error(f"Error while saving model {e}")
            raise CustomException("Failed to save model" ,  e)
    #running the model training pipeline using MLflow for experiment tracking
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting our Model Training pipeline")

                logger.info("Starting our MLFLOW experimentation")

                logger.info("Logging the training and testing datset to MLFLOW")
                mlflow.log_artifact(self.train_path , artifact_path="datasets")
                mlflow.log_artifact(self.test_path , artifact_path="datasets")

                X_train, X_test, y_train, y_test,scaler =self.load_and_split_data()
                # 1. Load or Generate Data (Replace this with your data loading logic)

                
                # 2. Initialize the ModelTuner
                self.tuner = ModelTuner(X_train, X_test, y_train, y_test)
                
                best_baseline_model_name,best_model, best_params,baseline_report=self.tuner.get_best_model()
                

                self.save_model(best_model, best_baseline_model_name,scaler)
            

                logger.info("Logging the model into MLFLOW")
                mlflow.log_artifact(self.model_output_path)

                logger.info("Logging Params and metrics to MLFLOW")
                mlflow.log_params(best_params)
                mlflow.log_metric("Best F1 Score", baseline_report.loc[best_baseline_model_name, 'F1 Score'])
                mlflow.log_metric("Best Accuracy", baseline_report.loc[best_baseline_model_name, 'Accuracy'])
                mlflow.log_metric("Best Precision", baseline_report.loc[best_baseline_model_name, 'Precision'])
                mlflow.log_metric("Best Recall", baseline_report.loc[best_baseline_model_name, 'Recall'])
                mlflow.sklearn.log_model(best_model, artifact_path="model")
                # Register the model
                mlflow.register_model(
                    "runs:/{}/model".format(mlflow.active_run().info.run_id),
                    best_baseline_model_name
                )
                

                logger.info("Model Training sucesfullly completed")

        except Exception as e:
            logger.error(f"Error in model training pipeline {e}")
            raise CustomException("Failed during model training pipeline" ,  e)
        

'''
#final running the model training class        
if __name__=="__main__":
    trainer = ModelTraining(PROCESSED_DIR,PROCESSED_DIR,MODEL_OUTPUT_PATH)
    trainer.run()
        
'''
    

            