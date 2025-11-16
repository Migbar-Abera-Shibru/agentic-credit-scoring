import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import os
from datetime import datetime
import json

class ModelRegistry:
    def __init__(self, registry_path: str = "./models/trained_models"):
        self.registry_path = registry_path
        self.registry_file = os.path.join(registry_path, "model_registry.json")
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from file"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save model registry to file"""
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, 
                      model_name: str, 
                      model_path: str, 
                      performance: Dict[str, float],
                      features: list,
                      metadata: Dict[str, Any] = None):
        """Register a new model in the registry"""
        model_info = {
            'model_path': model_path,
            'performance': performance,
            'features': features,
            'metadata': metadata or {},
            'registered_at': datetime.now().isoformat(),
            'version': f"v{len(self.registry) + 1}.0"
        }
        
        self.registry[model_name] = model_info
        self._save_registry()
        print(f"Model '{model_name}' registered successfully")
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model information from registry"""
        return self.registry.get(model_name)
    
    def load_model(self, model_name: str):
        """Load a model from the registry"""
        if model_name not in self.registry:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_path = self.registry[model_name]['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return joblib.load(model_path)
    
    def list_models(self) -> Dict[str, Any]:
        """List all registered models"""
        return self.registry
    
    def get_best_model(self, metric: str = 'auc_roc') -> str:
        """Get the best performing model based on a metric"""
        if not self.registry:
            return None
        
        best_model = None
        best_score = -1
        
        for model_name, info in self.registry.items():
            score = info['performance'].get(metric, -1)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model
    
    def update_model_performance(self, model_name: str, performance: Dict[str, float]):
        """Update model performance metrics"""
        if model_name in self.registry:
            self.registry[model_name]['performance'].update(performance)
            self.registry[model_name]['last_updated'] = datetime.now().isoformat()
            self._save_registry()
        else:
            raise ValueError(f"Model '{model_name}' not found in registry")