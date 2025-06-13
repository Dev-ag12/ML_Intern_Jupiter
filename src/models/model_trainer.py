import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.encoder = LabelEncoder()
        self.feature_importance = None
        
    def prepare_data(self, df):
        """Prepare data for modeling."""
        # Separate features and target
        X = df.drop(columns=['target'])
        y = df['target']
        
        # Encode target variable
        y_encoded = self.encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=self.random_state, stratify=y_encoded
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the model with SMOTE for handling class imbalance."""
        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ('smote_tomek', SMOTETomek(random_state=self.random_state)),
            ('classifier', RandomForestClassifier(class_weight='balanced_subsample', random_state=self.random_state))
        ])
        
        # Define parameter grid
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1
        )
        
        # Train model
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance."""
        # Predict
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # Get class-wise metrics
        recall = recall_score(
            y_test, y_pred, average=None
        )
        recall_dict = dict(zip(self.encoder.classes_, recall))
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'class_wise_recall': recall_dict
        }
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from the model."""
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.named_steps['classifier'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            return self.feature_importance
        return None
