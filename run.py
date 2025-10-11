#from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining

from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml, load_data
from config.buildconfig import generate_config
from config.paths_config import *
import pandas as pd

                
logger = get_logger(__name__)
logger.info("Training Pipeline Started")

##to build the config file from input dataset whcih contains categorical and numerical columns
generate_config(pd.read_csv(RAW_FILE_PATH))
logger.info("Config file generated successfully")
 
    
FEATURES_PATH =  'config/features.yaml'

print("Data Processing Started...",RAW_FILE_PATH)
                
processor = DataProcessor(config_path=CONFIG_PATH,features_path=FEATURES_PATH )
processor.run()

logger.info("Data Processing Completed...")
logger.info("Model Training Started...")


trainer = ModelTraining(PROCESSED_DIR,PROCESSED_DIR,MODEL_OUTPUT_PATH)
trainer.run()

logger.info("Model Training Completed...")
            
    
       