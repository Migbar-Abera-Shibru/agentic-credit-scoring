# src/credit_scoring_model.py - FINAL WORKING VERSION
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os
from typing import Dict, Any

class CreditScoringModel:
    def __init__(self, model_path: str = "./models/trained_models"):
        self.model_path = model_path
        self.model = None
        self.feature_importance = None
        
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, sample_size: int = None) -> Dict[str, Any]:
        """Train credit scoring model with sampling option"""
        print("Training credit scoring model...")
        
        # Use sampling if specified (for large datasets)
        if sample_size and sample_size < len(X_train):
            print(f"Using {sample_size:,} samples from {len(X_train):,} total")
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_train = X_train.iloc[indices]
            y_train = y_train.iloc[indices]
        
        # Convert to numpy arrays for stability
        X_array = X_train.values.astype(np.float32)
        y_array = y_train.values.astype(np.int32)
        
        print(f"Training on {X_array.shape[0]:,} samples with {X_array.shape[1]} features")
        print(f"Default rate: {y_array.mean()*100:.2f}%")
        
        # Train/validation split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_array, y_array, test_size=0.2, random_state=42, stratify=y_array
        )
        
        # Try models in order of complexity
        models_to_try = [
            ('logistic', LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)),
            ('random_forest', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
        ]
        
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models_to_try:
            print(f"  Training {name}...")
            
            try:
                model.fit(X_tr, y_tr)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                auc_score = roc_auc_score(y_val, y_pred_proba)
                
                results[name] = {
                    'model': model,
                    'auc_score': auc_score
                }
                
                print(f"    {name} - AUC: {auc_score:.4f}")
                
                if auc_score > best_score:
                    best_score = auc_score
                    best_model = model
                    
            except Exception as e:
                print(f"    {name} failed: {e}")
                continue
        
        if best_model is None:
            raise ValueError("All models failed to train")
        
        # Train final model on all data
        print(f"Training final model on all data...")
        self.model = best_model
        self.model.fit(X_array, y_array)
        
        print(f"🏆 Final model AUC: {best_score:.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n📊 Top 5 Most Important Features:")
            for i, row in self.feature_importance.head(5).iterrows():
                print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        return results
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of default"""
        if self.model is None:
            raise ValueError("No model has been trained")
        
        X_array = X.values.astype(np.float32)
        return self.model.predict_proba(X_array)[:, 1]
    
    def save_model(self, filename: str = "credit_scoring_model.pkl"):
        """Save the trained model"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        model_path = os.path.join(self.model_path, filename)
        joblib.dump({
            'model': self.model,
            'feature_importance': self.feature_importance
        }, model_path)
        
        print(f"Credit scoring model saved to {model_path}")