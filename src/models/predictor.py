"""
Student Risk Predictor
Machine learning models for predicting at-risk students
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import *


class StudentRiskPredictor:
    """
    Train and deploy machine learning models for student risk prediction
    """
    
    def __init__(self, random_state=RANDOM_SEED):
        """
        Initialize the predictor with multiple ML models
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {
            'logistic': LogisticRegression(random_state=random_state, max_iter=1000),
            'random_forest': RandomForestClassifier(
                n_estimators=N_ESTIMATORS, 
                random_state=random_state
            ),
            'gradient_boost': GradientBoostingClassifier(random_state=random_state)
        }
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, df):
        """
        Prepare data for training
        
        Args:
            df (pd.DataFrame): Feature DataFrame with 'student_id' and 'at_risk' columns
            
        Returns:
            tuple: (X_scaled, y) - Scaled features and target variable
        """
        print("Preparing data for training...")
        
        # Remove non-feature columns
        X = df.drop(['student_id', 'at_risk'], axis=1)
        y = df['at_risk']
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill any remaining NaN values after conversion
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        self.feature_names = X.columns.tolist()
        
        print(f"Data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        print(f"Class distribution: At-risk={y.sum()} ({y.mean()*100:.1f}%), Normal={len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")
        
        return X_scaled, y
    
    def train_models(self, X, y, test_size=TEST_SIZE, cv_folds=CV_FOLDS):
        """
        Train and evaluate multiple models
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of data for testing
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            dict: Dictionary of results for each model
        """
        print(f"\nTraining models with {cv_folds}-fold cross-validation...")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples\n")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=cv_folds, scoring='roc_auc'
            )
            
            # Test predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_auc': auc_score,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            print(f"  Test AUC: {auc_score:.4f}")
            print()
        
        # Select best model based on test AUC
        self.best_model_name = max(results.keys(), key=lambda x: results[x]['test_auc'])
        self.best_model = results[self.best_model_name]['model']
        
        print("=" * 60)
        print(f"✓ Best model: {self.best_model_name} (Test AUC: {results[self.best_model_name]['test_auc']:.4f})")
        print("=" * 60)
        
        return results
    
    def predict_risk(self, X):
        """
        Predict risk scores for new students
        
        Args:
            X (pd.DataFrame or np.array): Feature matrix
            
        Returns:
            np.array: Risk probability scores
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet! Please run train_models() first.")
        
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            # Remove non-numeric columns if they exist
            if 'student_id' in X.columns:
                X = X.drop('student_id', axis=1)
            if 'at_risk' in X.columns:
                X = X.drop('at_risk', axis=1)
        
        # Handle missing values
        X = pd.DataFrame(X).fillna(pd.DataFrame(X).median())
        
        # Ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill any remaining NaN values
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        predictions = self.best_model.predict_proba(X_scaled)[:, 1]
        
        return predictions
    
    def get_feature_importance(self):
        """
        Get feature importance from best model (if available)
        
        Returns:
            pd.DataFrame: DataFrame with features and their importance scores
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        elif hasattr(self.best_model, 'coef_'):
            # For logistic regression, use absolute coefficients
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(self.best_model.coef_[0])
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            print("Feature importance not available for this model type")
            return None
    
    def save_model(self, directory=MODELS_DIR, model_name='best_model'):
        """
        Save the trained model and scaler
        
        Args:
            directory (str): Directory to save the model
            model_name (str): Base name for the model files
        """
        if self.best_model is None:
            raise ValueError("No trained model to save!")
        
        os.makedirs(directory, exist_ok=True)
        
        model_path = os.path.join(directory, f'{model_name}.pkl')
        scaler_path = os.path.join(directory, f'{model_name}_scaler.pkl')
        features_path = os.path.join(directory, f'{model_name}_features.pkl')
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_names, features_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        print(f"Features saved to: {features_path}")
        
        return model_path
    
    def load_model(self, directory=MODELS_DIR, model_name='best_model'):
        """
        Load a trained model and scaler
        
        Args:
            directory (str): Directory containing the model files
            model_name (str): Base name for the model files
        """
        model_path = os.path.join(directory, f'{model_name}.pkl')
        scaler_path = os.path.join(directory, f'{model_name}_scaler.pkl')
        features_path = os.path.join(directory, f'{model_name}_features.pkl')
        
        self.best_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        
        print(f"Model loaded from: {model_path}")
        print(f"Model type: {type(self.best_model).__name__}")
        
        return self.best_model


if __name__ == "__main__":
    # Test the predictor
    from src.data.data_generator import LMSDataGenerator
    from src.features.feature_engineer import FeatureEngineer
    
    # Generate data
    generator = LMSDataGenerator(n_students=500)
    raw_data = generator.generate_student_data(n_days=60)
    
    # Create features
    engineer = FeatureEngineer()
    features = engineer.create_features(raw_data)
    
    # Train models
    predictor = StudentRiskPredictor()
    X, y = predictor.prepare_data(features)
    results = predictor.train_models(X, y)
    
    # Show feature importance
    importance = predictor.get_feature_importance()
    if importance is not None:
        print("\nTop 10 Most Important Features:")
        print(importance.head(10))