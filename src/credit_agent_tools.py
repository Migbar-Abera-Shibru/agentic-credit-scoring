# src/credit_agent_tools.py - VERIFY IMPORTS
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import joblib

class CreditAgentTools:
    def __init__(self, feature_importance: pd.DataFrame = None):
        self.feature_importance = feature_importance
        self.risk_thresholds = {
            'low': 0.1,
            'medium': 0.3, 
            'high': 0.7
        }
        
        self.feature_descriptions = {
            'RevolvingUtilizationOfUnsecuredLines': 'Credit card utilization rate (ideal: <0.3)',
            'age': 'Age of borrower (ideal: 25-65)',
            'NumberOfTime30-59DaysPastDueNotWorse': '30-59 days late payments (ideal: 0)',
            'DebtRatio': 'Monthly debt payments / Monthly income (ideal: <0.36)',
            'MonthlyIncome': 'Monthly income amount',
            'NumberOfOpenCreditLinesAndLoans': 'Total open credit lines',
            'NumberOfTimes90DaysLate': '90+ days late payments (high risk if >0)',
            'NumberRealEstateLoansOrLines': 'Mortgage and real estate loans',
            'NumberOfTime60-89DaysPastDueNotWorse': '60-89 days late payments (ideal: 0)',
            'NumberOfDependents': 'Number of dependents'
        }
    
    def analyze_risk_factors(self, features: Dict[str, Any], probability: float) -> List[str]:
        """Analyze specific risk factors for a borrower"""
        risk_factors = []
        
        # Credit utilization analysis
        utilization = features.get('RevolvingUtilizationOfUnsecuredLines', 0)
        if utilization > 0.8:
            risk_factors.append(f"Very high credit utilization ({utilization:.1%})")
        elif utilization > 0.5:
            risk_factors.append(f"High credit utilization ({utilization:.1%})")
        
        # Late payments analysis
        late_90_days = features.get('NumberOfTimes90DaysLate', 0)
        if late_90_days > 0:
            risk_factors.append(f"Has {late_90_days} serious late payment(s) (90+ days)")
        
        late_60_days = features.get('NumberOfTime60-89DaysPastDueNotWorse', 0)
        if late_60_days > 0:
            risk_factors.append(f"Has {late_60_days} late payment(s) (60-89 days)")
        
        late_30_days = features.get('NumberOfTime30-59DaysPastDueNotWorse', 0)
        if late_30_days > 0:
            risk_factors.append(f"Has {late_30_days} late payment(s) (30-59 days)")
        
        # Debt ratio analysis
        debt_ratio = features.get('DebtRatio', 0)
        if debt_ratio > 0.5:
            risk_factors.append(f"High debt ratio ({debt_ratio:.1%})")
        
        # Age analysis
        age = features.get('age', 0)
        if age < 25:
            risk_factors.append("Young applicant with limited credit history")
        elif age > 70:
            risk_factors.append("Older applicant near retirement")
        
        # Income analysis
        income = features.get('MonthlyIncome', 0)
        if income < 2000:
            risk_factors.append("Low monthly income")
        
        return risk_factors
    
    def generate_recommendations(self, risk_factors: List[str], probability: float) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if probability > 0.7:
            recommendations.append("Consider requiring a co-signer")
            recommendations.append("Suggest smaller loan amount")
        
        if "credit utilization" in str(risk_factors).lower():
            recommendations.append("Recommend paying down credit card balances")
            recommendations.append("Suggest credit utilization below 30%")
        
        if "late payment" in str(risk_factors).lower():
            recommendations.append("Focus on improving payment history")
            recommendations.append("Consider automatic payment setup")
        
        if "debt ratio" in str(risk_factors).lower():
            recommendations.append("Recommend debt consolidation")
            recommendations.append("Suggest increasing income or reducing expenses")
        
        if not recommendations:
            recommendations.append("Application meets standard criteria")
        
        return recommendations
    
    def simulate_scenario(self, features: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Simulate what-if scenarios"""
        modified_features = features.copy()
        
        if "income" in scenario.lower():
            if "20%" in scenario:
                current_income = features.get('MonthlyIncome', 0)
                modified_features['MonthlyIncome'] = current_income * 1.2
                return {"scenario": "20% income increase", "modified_feature": "MonthlyIncome", "modified_value": current_income * 1.2}
        
        elif "debt" in scenario.lower():
            if "reduce" in scenario.lower():
                current_debt_ratio = features.get('DebtRatio', 0)
                modified_features['DebtRatio'] = max(0, current_debt_ratio * 0.7)
                return {"scenario": "30% debt reduction", "modified_feature": "DebtRatio", "modified_value": current_debt_ratio * 0.7}
        
        elif "utilization" in scenario.lower():
            current_utilization = features.get('RevolvingUtilizationOfUnsecuredLines', 0)
            modified_features['RevolvingUtilizationOfUnsecuredLines'] = max(0, current_utilization * 0.5)
            return {"scenario": "50% credit utilization reduction", "modified_feature": "RevolvingUtilizationOfUnsecuredLines", "modified_value": current_utilization * 0.5}
        
        return {"scenario": "Unknown scenario", "modified_feature": None, "modified_value": None}
    
    def get_feature_explanations(self, features: Dict[str, Any]) -> List[str]:
        """Generate feature-based explanations"""
        explanations = []
        
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(3)
            for _, row in top_features.iterrows():
                feature = row['feature']
                importance = row['importance']
                value = features.get(feature, 'N/A')
                description = self.feature_descriptions.get(feature, feature)
                explanations.append(f"**{feature}** (importance: {importance:.3f}): {value} - {description}")
        
        return explanations