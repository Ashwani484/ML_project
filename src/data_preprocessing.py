import os
import pandas as pd
import numpy as np
import joblib
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer  # Imported for handling missing values
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

# class for Data Preprocessing
class DataProcessor:
    """
    This class handles all data preprocessing steps 
    """

    # Constructor updated to accept paths for both config and features YAML files
    def __init__(self, config_path, features_path):

        
        self.processed_dir = PROCESSED_DIR
        
        # Load configurations from both YAML files
        self.config = read_yaml(config_path)
        self.features_config = read_yaml(features_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocess_data(self, df):
        """
        Applies cleaning and imputation.
        """
        try:
            logger.info("Data preprocess Started for Data Cleaning and Imputation step")

            # --- 1. Drop Columns based on config.yaml ---
            features_to_drop = self.config.get('feature_engineering', {}).get('features_to_drop', [])
            df.drop(columns=features_to_drop, inplace=True, errors='ignore')
            df.drop_duplicates(inplace=True)
            logger.info(f"Dropped columns done: {features_to_drop}")

            # --- Get column lists from features.yaml ---
            cat_cols_from_yaml = self.features_config["data_processing"]["categorical_columns"]
            num_cols_from_yaml = self.features_config["data_processing"]["numerical_columns"]

            # Filter columns to only those that exist in the DataFrame
            num_cols = [col for col in num_cols_from_yaml if col in df.columns]
            cat_cols = [col for col in cat_cols_from_yaml if col in df.columns]

            # --- 2. Handle Missing Values ---
            logger.info("Handling missing values...")
            if num_cols: # Only run if num_cols is not empty
                num_imputer_strategy = self.config['missing_value_handling']['numerical_imputation']['strategy']
                num_imputer = SimpleImputer(strategy=num_imputer_strategy)
                df[num_cols] = num_imputer.fit_transform(df[num_cols])
                logger.info(f"Numerical columns imputed using '{num_imputer_strategy}' strategy and if any missing values",df.isnull().sum())

            if cat_cols: # Only run if cat_cols is not empty
                cat_imputer_strategy = self.config['missing_value_handling']['categorical_imputation']['strategy']
                cat_imputer = SimpleImputer(strategy=cat_imputer_strategy)
                df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
                logger.info(f"Categorical columns imputed using '{cat_imputer_strategy}' strategy.if any missing values",df.isnull().sum())

            # --- 3. Skewness Handling ---
            logger.info("Handling Skewness...")
            skew_threshold = self.config["data_ingestion"]["skewness_threshold"]
            # Check for skewness only on existing numerical columns
            skewness = df[num_cols].skew()
            for column in skewness[skewness > skew_threshold].index:
                logger.info(f"Applying log1p transformation to skewed column: {column}")
                df[column] = np.log1p(df[column])

            return df

        except Exception as e:
            logger.error(f"Error during preprocessing step: {e}", exc_info=True)
            raise CustomException(f"Error while preprocessing data: {e}", e)
        
            # --- 3. Label Encoding ---
    def encoding(self,df):
        try:
            X = df.drop(columns=self.config["output_feature"])
            y = df[self.config["output_feature"]]
            
            # --- 3. Feature Encoding based on config.yaml ---
            logger.info("Applying feature encoding based on 'config.yaml' specifications")

            # Read the encoding configuration from the YAML file
            # Defaults to 'LabelEncoding' if the key is not found
            encoding_config = self.config.get('encoding', {})
            encoder_type = encoding_config.get('encoder_type', 'LabelEncoding')

            # Get the list of all categorical columns from the features configuration
            all_cat_cols = self.features_config["data_processing"]["categorical_columns"]
            # Filter for only those columns that currently exist in the DataFrame
            cat_cols_to_encode = [col for col in all_cat_cols if col in X.columns]


            # --- Conditionally Apply the Specified Encoding Technique ---

            if encoder_type == 'OneHotEncoder' and cat_cols_to_encode:
                logger.info(f"Applying One-Hot Encoding to columns: {cat_cols_to_encode}")
                
                # Use pandas.get_dummies to convert categorical variables into dummy/indicator variables
                X = pd.get_dummies(X, columns=cat_cols_to_encode, drop_first=True)
                
                logger.info("One-Hot Encoding completed successfully.",X.dtypes)

            elif encoder_type == 'LabelEncoder' and cat_cols_to_encode:
                logger.info(f"Applying Label Encoding to columns: {cat_cols_to_encode}")
                
                label_encoder = LabelEncoder()
                mappings = {}
                
                for col in cat_cols_to_encode:
                    X[col] = label_encoder.fit_transform(X[col])
                    mappings[col] = {label: int(code) for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
                
                logger.info("LabelEncoding Mappings created successfully.",X.dtypes)
                # Consider saving the 'mappings' dictionary to a file for later use (e.g., inverting the transform)

            else:
                if not cat_cols_to_encode:
                    logger.warning("No categorical columns were found in the DataFrame to encode.")
                else:
                    logger.error(f"'{encoder_type}' is not a recognized encoder type in the configuration. No encoding was applied.")

            
            return X, y

        except Exception as e:
            logger.error(f"Error during preprocess step: {e}")
            raise CustomException(f"Error while preprocessing data: {e}", e)

    

    def feature_selection(self, X,y):
        """Selects top features using RandomForestClassifier based on importance."""
        try:
            logger.info("Starting Feature selection step")
        

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            feature_importance = model.feature_importances_
        

            feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})
            top_features_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)

            # Corrected path to no_of_features from config.yaml
            num_features_to_select = self.config["data_ingestion"]["no_of_features"]
            selected_features = top_features_importance_df["feature"].head(num_features_to_select).values

            final_df = X[selected_features].copy()

            # 2. Add the target Series 'y' as a new column to this DataFrame.
        
            target_column_name = self.config.get("output_feature", "target") # Fallback to 'target'
            final_df[target_column_name] = y.values
        

            logger.info(f"Feature selection completed successfully and selected features:-->>{final_df.columns}")
            return final_df

        except Exception as e:
            logger.error(f"Error during feature selection step: {e}")
            raise CustomException(f"Error while feature selection: {e}", e)
        

    ''' 
    # chi2 feature selection method (not used)  
    def feature_selection(self,X,y):
        try:
            X_train , _, y_train , _ = train_test_split(self.X,self.y , test_size=0.2 , random_state=42)

            X_cat = X_train.select_dtypes(include=['int64' , 'float64'])
            chi2_selector = SelectKBest(score_func=chi2 , k="all")
            chi2_selector.fit(X_cat,y_train)

            chi2_scores = pd.DataFrame({
                    'Feature' : X_cat.columns,
                    "Chi2 Score" : chi2_selector.scores_
                }).sort_values(by='Chi2 Score' , ascending=False)
            
            top_features = chi2_scores.head(5)["Feature"].tolist()
            self.selected_features = top_features
            logger.info(f"Selected features are : {self.selected_features}")

            self.X = self.X[self.selected_features]
            logger.info("Feature selection done..")
        
        except Exception as e:
            logger.error(f"Error while feature selection data {e}")
            raise CustomException("Failed to feature selection data")    '''
        

    def balance_data(self, df: pd.DataFrame):

        try:
            logger.info("Handling imbalanced data with SMOTE")

            # Get target column from config
            target_column = self.config.get("output_feature")
            if not target_column or target_column not in df.columns:
                logger.warning(f"Target column '{target_column}' not found. Skipping balancing.")
                return df, None, None, None, None
            

            # Split features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            logger.info(f"Before BalanceData-->>{y.value_counts()}")

            # Train-test split (stratified to preserve imbalance in test set)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42)

            # Apply SMOTE only on training data
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            # Rebuild balanced training DataFrame
            train_balanced_df = pd.DataFrame(X_train_res, columns=X_train.columns)
            train_balanced_df[target_column] = y_train_res
            

            # Rebuild untouched testing DataFrame
            test_df = X_test.copy()
            test_df[target_column] = y_test.values
            test_df.reset_index(drop=True, inplace=True)

            logger.info("Training data balanced successfully with SMOTE")
            logger.info(f"After BalanceData-->>{train_balanced_df[target_column].value_counts()}")

            # Return both balanced train and untouched test sets
            return train_balanced_df, test_df

        except Exception as e:
            logger.error(f"Error during balancing data step: {e}", exc_info=True)
            raise CustomException(f"Error while balancing data: {e}", e)

            

    def save_data(self, train_df,test_df, file_path):
        """Saves the processed dataframe to a CSV file."""
        try:
            logger.info(f"Saving data to {file_path}")
            joblib.dump(train_df , os.path.join(file_path , 'train_df.pkl'))
            joblib.dump(test_df , os.path.join(file_path , 'test_df.pkl'))
            #train_df.to_csv(os.path.join(file_path , 'train_df.csv'))
            #test_df.to_csv(os.path.join(file_path , 'test_df.csv'))
            
            
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error during saving data step: {e}")
            raise CustomException(f"Error while saving data: {e}", e)

    def run(self):
        """Main orchestrator method to run the entire preprocessing pipeline."""
        try:
            logger.info("Loading data from RAW directory")
            df = load_data(RAW_FILE_PATH)
            logger.info("Data loaded successfully")
            logger.info("--- Processing Training Data ---")

            processed_df = self.preprocess_data(df)
            df_X,df_y=self.encoding(processed_df)
            df_feature_selection=self.feature_selection(df_X,df_y)
            train_df, test_df=train_test_split(df_feature_selection, test_size=0.2, random_state=42)
            #train_balance_df, test_df = self.balance_data(df_feature_selection)  # Balance only the training data
            self.save_data(train_df, test_df, PROCESSED_DIR)

            logger.info("Data processing completed successfully")
        except Exception as e:
            logger.error(f"Error during preprocessing pipeline: {e}")
            raise CustomException(f"Error during data preprocessing pipeline: {e}", e)

# To run this class, you must pass the paths to both YAML files.
if __name__ == "__main__":
    # Assuming FEATURES_PATH is defined in your paths_config or here
    FEATURES_PATH =  'config/features.yaml'
    print("Data Processing Started...",RAW_FILE_PATH)
    processor = DataProcessor(
        
        config_path=CONFIG_PATH,
        features_path=FEATURES_PATH  # Pass the new path here
    )
    processor.run()


