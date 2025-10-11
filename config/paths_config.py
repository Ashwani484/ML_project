import os
import glob
from pathlib import Path

########################### DATA INGESTION #########################
folder_path = Path(r"raw_dataset")
file_name = next(folder_path.glob("*")).name


print("Input Raw_Dataset file name:", file_name)
RAW_FILE_PATH = os.path.join(folder_path,file_name)
print("Raw file path is:", RAW_FILE_PATH)


RAW_DIR = "artifacts//raw"

TRAIN_FILE_PATH = os.path.join(RAW_DIR,"train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR,"test.csv")

CONFIG_PATH = "config//config.yaml"


######################## DATA PROCESSING ########################

PROCESSED_DIR = "artifacts/processed"
#PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR,"processed_train.csv")
#PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR,"processed_test.csv")


####################### MODEL TRAINING #################

path = Path(r"artifacts/models")
files = list(path.glob("*"))
if files:
    file_names = files[0].name
    print("Input Raw_Dataset file name:", file_names)
    MODEL_OUTPUT_PATH = os.path.join(path, file_names)
else:
    print(" The Model folder is empty:", path)
    MODEL_OUTPUT_PATH = "artifacts/models"

#MODEL_OUTPUT_PATH="artifacts/models/random_forest.pkl"






