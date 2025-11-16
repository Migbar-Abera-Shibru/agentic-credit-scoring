# src/credit_data_processor.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class CreditDataProcessor:
    def __init__(self, target_column: str = "SeriousDlqin2yrs", id_column: str = "Unnamed: 0"):
        self.target_column = target_column
        self.id_column = id_column
        self.feature_columns = None
        self.feature_descriptions = {
            'RevolvingUtilizationOfUnsecuredLines': 'Credit card utilization rate (0-1)',
            'age': 'Age of borrower',
            'NumberOfTime30-59DaysPastDueNotWorse': 'Number of 30-59 days past due incidents',
            'DebtRatio': 'Monthly debt payments / Monthly income',
            'MonthlyIncome': 'Monthly income amount',
            'NumberOfOpenCreditLinesAndLoans': 'Number of open credit lines and loans',
            'NumberOfTimes90DaysLate': 'Number of 90+ days late payments',
            'NumberRealEstateLoansOrLines': 'Number of mortgage and real estate loans',
            'NumberOfTime60-89DaysPastDueNotWorse': 'Number of 60-89 days past due incidents',
            'NumberOfDependents': 'Number of dependents'
        }
    
    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare credit scoring data"""
        print("Loading credit scoring data...")
        
        # Load training data
        train_df = pd.read_csv(train_path)
        print(f"Training data: {train_df.shape}")
        
        # Load test data - remove target column if it exists
        test_df = pd.read_csv(test_path)
        if self.target_column in test_df.columns:
            test_df = test_df.drop(columns=[self.target_column])
        print(f"Test data: {test_df.shape}")
        
        return train_df, test_df
    
    def preprocess_credit_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess credit scoring data"""
        print("Preprocessing credit data...")
        
        # Handle missing values
        train_processed = self.handle_missing_values(train_df)
        test_processed = self.handle_missing_values(test_df)
        
        # Define feature columns 
        all_columns = train_processed.columns.tolist()
        self.feature_columns = [col for col in all_columns 
                               if col not in [self.target_column, self.id_column]]
        
        print(f"Using {len(self.feature_columns)} features: {self.feature_columns}")
        
        return train_processed, test_processed
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in credit data"""
        df_clean = df.copy()
        
        # MonthlyIncome 
        if 'MonthlyIncome' in df_clean.columns and df_clean['MonthlyIncome'].isnull().any():
            median_income = df_clean['MonthlyIncome'].median()
            df_clean['MonthlyIncome'] = df_clean['MonthlyIncome'].fillna(median_income)
            print(f"Filled {df_clean['MonthlyIncome'].isnull().sum()} missing MonthlyIncome with median: {median_income:.2f}")
        
        # NumberOfDependents 
        if 'NumberOfDependents' in df_clean.columns and df_clean['NumberOfDependents'].isnull().any():
            mode_dependents = df_clean['NumberOfDependents'].mode()[0] if not df_clean['NumberOfDependents'].mode().empty else 0
            df_clean['NumberOfDependents'] = df_clean['NumberOfDependents'].fillna(mode_dependents)
            print(f"Filled {df_clean['NumberOfDependents'].isnull().sum()} missing NumberOfDependents with mode: {mode_dependents}")
        
        return df_clean
    
    def get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description of a feature"""
        return self.feature_descriptions.get(feature_name, feature_name)