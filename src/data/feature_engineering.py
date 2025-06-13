import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class FeatureEngineer:
    def __init__(self):
        self.numeric_features = [
            'age', 'monthly_income', 'monthly_emi_outflow', 'current_outstanding',
            'credit_utilization_ratio', 'repayment_history_score', 'dpd_last_3_months',
            'num_hard_inquiries_last_6m', 'recent_credit_card_usage',
            'recent_loan_disbursed_amount', 'total_credit_limit',
            'months_since_last_default', 'num_open_loans'
        ]
        
        self.categorical_features = ['gender', 'location']
        
        self.target = 'target_credit_score_movement'
    
    def create_preprocessor(self):
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, self.numeric_features),
            ('cat', categorical_transformer, self.categorical_features)
        ])
        
        return preprocessor
    
    def create_derived_features(self, df):
        df['income_to_emi_ratio'] = df['monthly_income'] / df['monthly_emi_outflow']
        df["recent_usage_rattio"]=df['recent_credit_card_usage'] / df['total_credit_limit']
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for modeling."""
        df = self.create_derived_features(df)
        
        preprocessor = self.create_preprocessor()
        
        X = df.drop(columns=[self.target])
        y = df[self.target]

        X_processed = preprocessor.fit_transform(X)
        
        return X_processed, y, preprocessor
    
    def get_feature_names(self, preprocessor):
        """Get feature names after preprocessing."""
        numeric_features = self.numeric_features
        categorical_features = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
            self.categorical_features))
        
        return numeric_features + categorical_features
