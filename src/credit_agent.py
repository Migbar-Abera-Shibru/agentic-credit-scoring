import pandas as pd  
import numpy as np
from typing import Dict, Any, List
from .credit_agent_tools import CreditAgentTools

class CreditAgent:
    def __init__(self, model, feature_importance: pd.DataFrame = None):
        self.model = model
        self.tools = CreditAgentTools(feature_importance)
    
    def process_query(self, features: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Process agentic queries with reasoning for credit decisions"""
        try:
            # Convert features to DataFrame for prediction
            features_df = self._features_to_dataframe(features)
            
            # Get base probability
            probability = float(self.model.predict_proba(features_df)[0, 1])
            
            # Initialize response
            response = {
                "probability": probability,
                "risk_level": self._get_risk_level(probability),
                "reasoning": [],
                "risk_factors": [],
                "recommendations": [],
                "tools_used": []
            }
            
            # Analyze risk factors
            risk_factors = self.tools.analyze_risk_factors(features, probability)
            response["risk_factors"] = risk_factors
            response["tools_used"].append("risk_analysis")
            
            # Generate reasoning steps
            response["reasoning"].append(f"Initial assessment: {probability:.1%} probability of serious delinquency")
            response["reasoning"].append(f"Risk classification: {response['risk_level']}")
            
            if risk_factors:
                response["reasoning"].append("Key risk factors identified:")
                for risk in risk_factors:
                    response["reasoning"].append(f"  • {risk}")
            
            # Feature explanations
            feature_explanations = self.tools.get_feature_explanations(features)
            if feature_explanations:
                response["reasoning"].append("Most influential factors:")
                for explanation in feature_explanations:
                    response["reasoning"].append(f"  • {explanation}")
            
            # Handle specific queries
            query_lower = query.lower()
            
            if "what if" in query_lower:
                scenario_result = self.tools.simulate_scenario(features, query)
                if scenario_result["modified_feature"]:
                    response["reasoning"].append(f"Scenario analysis: {scenario_result['scenario']}")
                    response["tools_used"].append("scenario_simulation")
            
            if "recommend" in query_lower or "improve" in query_lower or "suggest" in query_lower:
                recommendations = self.tools.generate_recommendations(risk_factors, probability)
                response["recommendations"] = recommendations
                response["reasoning"].append("Personalized recommendations generated")
            
            if "explain" in query_lower:
                response["reasoning"].append("Detailed explanation provided based on feature importance and risk factors")
            
            return response
            
        except Exception as e:
            # Return error information for debugging
            return {
                "probability": 0.0,
                "risk_level": "Error",
                "reasoning": [f"Error processing query: {str(e)}"],
                "risk_factors": [],
                "recommendations": [],
                "tools_used": ["error_handling"]
            }
    
    def _features_to_dataframe(self, features: Dict[str, Any]):
        """Convert features dict to DataFrame with correct column order"""
        # Expected features 
        expected_features = [
            'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
            'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 
            'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
        ]
        
        # Create DataFrame with correct column order
        features_ordered = {feature: features.get(feature, 0) for feature in expected_features}
        return pd.DataFrame([features_ordered])
    
    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level"""
        if probability < 0.1:
            return "Low Risk"
        elif probability < 0.3:
            return "Medium Risk"
        elif probability < 0.7:
            return "High Risk"
        else:
            return "Very High Risk"