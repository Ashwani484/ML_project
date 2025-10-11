import pandas as pd
import yaml
import glob
import os
from config.paths_config import *



def generate_config(raw_df):
    
    # Step 1: Load dataset
    df= pd.DataFrame(raw_df)

    # Step 2: Separate categorical and numerical columns
    categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Step 3: Build config dictionary
    config = {
        
        "data_processing": {
            "categorical_columns": categorical_columns,
            "numerical_columns": numerical_columns,
            
        }
    }

    # Step 4: Save to YAML file
    with open("config/features.yaml", "w") as file:
        yaml.dump(config, file, sort_keys=False, default_flow_style=False)

    print("✅ config.yml generated successfully!")




