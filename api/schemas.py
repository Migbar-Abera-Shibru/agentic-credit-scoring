# api/schemas.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class PredictionInput(BaseModel):
    features: Dict[str, Any] = Field(..., example={
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 35,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.3,
        "MonthlyIncome": 5000,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1
    })

class PredictionOutput(BaseModel):
    probability: float = Field(..., ge=0, le=1, example=0.15)

class AgentInput(BaseModel):
    features: Dict[str, Any] = Field(..., example={
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 35,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.3,
        "MonthlyIncome": 5000,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1
    })
    query: str = Field(..., example="Explain my risk factors and suggest improvements")

class AgentOutput(BaseModel):
    probability: float = Field(..., ge=0, le=1, example=0.15)
    risk_level: str = Field(..., example="Low Risk")
    reasoning: List[str] = Field(..., example=[
        "Initial assessment: 15.0% probability of serious delinquency",
        "Risk classification: Low Risk",
        "Key risk factors identified:",
        "  • High credit utilization (50.0%)"
    ])
    risk_factors: List[str] = Field(..., example=["High credit utilization (50.0%)"])
    recommendations: List[str] = Field(..., example=["Recommend paying down credit card balances"])
    tools_used: List[str] = Field(..., example=["risk_analysis", "scenario_simulation"])