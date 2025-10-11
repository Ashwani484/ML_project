import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.datasets import make_classification # For generating sample data
from sklearn.model_selection import train_test_split
from src.logger import get_logger


from config.paths_config import *
logger = get_logger(__name__)
# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

class ModelTuner:
    """
    A class to handle model evaluation and hyperparameter tuning for various classification models.
    """
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initializes the ModelTuner with training and testing data.
        
        Args:
            X_train: Training feature data
            y_train: Training target data
            X_test: Testing feature data
            y_test: Testing target data
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = self._get_models()
        self.params = self._get_params()

    def _get_models(self):
        """Returns a dictionary of classification models."""
        return {
            "LogisticRegression": LogisticRegression(),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier(),
            "SVM": SVC(),
            #"XGBoost": XGBClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "GradientBoosting": GradientBoostingClassifier(),
            "LightGBM": lgb.LGBMClassifier()
        }

    def _get_params(self):
        """Returns a dictionary of hyperparameters for each model."""
        return {
            "LogisticRegression": {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': np.logspace(-4, 4, 20),
                'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
                'max_iter': [100, 50, 200, 300]
            },
            "DecisionTree": {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [None, 20, 30, 40,50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [ 2, 4, 6]
            },
            "RandomForest": {
                'n_estimators': [50, 100, 200],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "SVM": {
                'C': [0.1, 1, 10, 100],
                'gamma': [1, 0.1, 0.01, 0.001],
                'kernel': ['rbf', 'poly', 'sigmoid']
            },
            '''
            "XGBoost": {
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9]
            },  '''


            "AdaBoost": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1],
                'algorithm': ['SAMME', 'SAMME.R']
            },
            "GradientBoosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9]
            },
            "LightGBM": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [20, 31, 50, 100],    
                'boosting_type': ['gbdt', 'dart', 'goss']
            }
        }
        
    def evaluate_models(self):
        """
        Fits and evaluates all models with default parameters.

        Returns:
            pd.DataFrame: A DataFrame containing the performance metrics for each model.
        """
        logger.info("Model Evaluation started")
        report = {}
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            logger.info(f"Evaluating {name}...")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            report[name] = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred, average='weighted'),
                'Recall': recall_score(self.y_test, y_pred, average='weighted'),
                'F1 Score': f1_score(self.y_test, y_pred, average='weighted')
            }
        print("="*60)
        
        metrics=pd.DataFrame(report).T.sort_values(by='F1 Score', ascending=False)
        logger.info(f"Evaluating {metrics}...")
        
        return metrics

    def tune_model(self, model_name, method='random', n_iter=6, cv=5, verbose=2):
        """
        Performs hyperparameter tuning for a specified model.

        Args:
            model_name (str): The name of the model to tune (must be a key in self.models).
            method (str): 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.
            n_iter (int): Number of parameter settings that are sampled for RandomizedSearchCV.
            cv (int): Number of cross-validation folds.
            verbose (int): Controls the verbosity of the search process.

        Returns:
            tuple: A tuple containing the best estimator and the best parameters found.
        """

        logger.info("Model Tunning Started...")

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")

        model = self.models[model_name]
        params = self.params[model_name]
        
        #print(f"\n--- Starting Hyperparameter Tuning for {model_name} using {method.capitalize()}SearchCV ---")
        
        searcher = None
        #if method == 'grid':
        #    searcher = GridSearchCV(estimator=model, param_grid=params, cv=cv, verbose=verbose, n_jobs=-1)
        if method == 'random':
            searcher = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=n_iter, cv=cv, verbose=verbose, n_jobs=-1, random_state=42)
        else:
            raise ValueError("Method must be 'grid' or 'random'.")
            
        searcher.fit(self.X_train, self.y_train)
        
        best_estimator = searcher.best_estimator_
        best_params = searcher.best_params_
        
        print(f"\nBest Parameters for {model_name}: {best_params}")

        logger.info((f"\nBest Parameters for {model_name}: {best_params}"))
        
        # Evaluate the best model on the test set
        y_pred = best_estimator.predict(self.X_test)
        print("\nPerformance of Tuned Model on Test Set:")

        print(classification_report(self.y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        print(f"{'='*60}\n")
        
        logger.info(f"The best estimator and Best parameter are as{model_name}:{best_estimator} :{best_params}")
        logger.info("Model Tunning completed")
        return best_estimator, best_params


    def get_best_model(self):
        """
        Identifies and returns the best model based on F1 Score from the baseline evaluation.

        Returns:
            str: The name of the best performing model.
        """
        
        # 3. Get Baseline Performance of all models
        print("--- Starting Baseline Model Evaluation ---")
        baseline_report = ModelTuner.evaluate_models(self)

        print("Baseline Model Performance Report:")
        print(baseline_report)
        print("="*60)

        # 4. Perform Hyperparameter Tuning on the best baseline model (or any model of choice)
        best_baseline_model_name = baseline_report.index[0]
        # Example: choosing the best performing model
        
        print("The Best Selected Model is--->>>",best_baseline_model_name)

        logger.info("The Best Selected Model is--->>>",best_baseline_model_name)


        # --- Example 1: Randomized Search CV on Random Forest ---
        # Randomized Search is often a good first step as it's faster than Grid Search
        rf_best_model, rf_best_params = ModelTuner.tune_model(self,
            model_name=best_baseline_model_name, 
            method='random',
            n_iter=10 # Number of combinations to try
        )

        

        print("The Best model parameters are as---->>>>",rf_best_model, rf_best_params )
        logger.info("The Best model parameters are as---->>>>",rf_best_model, rf_best_params )

        logger.info("The ModelTunning for Classification problem completed and the Best model is declered")

        return best_baseline_model_name, rf_best_model, rf_best_params,baseline_report






    '''
    # --- Example 2: Grid Search CV on Logistic Regression ---
    # Grid Search is more exhaustive and can be used to fine-tune further
    lr_best_model, lr_best_params = tuner.tune_model(
        model_name="Logistic Regression", 
        method='grid'
    )

    '''
    






