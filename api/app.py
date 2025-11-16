# api/app.py - FIXED PATH HANDLING
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
import sys
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from api.schemas import PredictionInput, PredictionOutput, AgentInput, AgentOutput
from config import settings

app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded model and agent
model = None
credit_agent = None

@app.on_event("startup")
async def startup_event():
    """Load model and agent on startup with safe path handling"""
    global model, credit_agent
    try:
        # Safe path construction
        if isinstance(settings.MODEL_PATH, Path):
            model_path = settings.MODEL_PATH / "credit_scoring_model.pkl"
        else:
            # Fallback: construct path manually
            model_path = Path(settings.MODEL_PATH) / "credit_scoring_model.pkl"
        
        print(f"🔧 Loading model from: {model_path}")
        print(f"🔧 Absolute path: {model_path.absolute()}")
        print(f"🔧 File exists: {model_path.exists()}")
        
        if model_path.exists():
            print("📦 Loading model data...")
            loaded_data = joblib.load(model_path)
            model = loaded_data['model']
            feature_importance = loaded_data.get('feature_importance')
            
            print(f"✅ Model type: {type(model).__name__}")
            
            # Test the model works
            test_features = [[0.5, 35, 0, 0.3, 5000, 5, 0, 1, 0, 1]]
            test_prob = model.predict_proba(test_features)[0, 1]
            print(f"✅ Model test prediction: {test_prob:.3f}")
            
            # Initialize credit agent
            print("🤖 Initializing credit agent...")
            from src.credit_agent import CreditAgent
            credit_agent = CreditAgent(model, feature_importance)
            
            print("✅ Agent initialized successfully")
            print("🚀 Agentic Credit Scoring API is ready!")
        else:
            print(f"❌ Model file not found at: {model_path}")
            print("💡 Make sure the path is correct and model exists")
            
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        import traceback
        traceback.print_exc()

@app.get("/")
async def root():
    return {
        "message": "Agentic Credit Scoring MaaS API", 
        "version": settings.API_VERSION,
        "status": "operational",
        "endpoints": {
            "documentation": "/docs",
            "health": "/health",
            "prediction": "/predict",
            "agent": "/agent"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "agent_loaded": credit_agent is not None
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_credit_risk(input_data: PredictionInput):
    """Predict credit risk probability"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert features to dataframe
        features_df = pd.DataFrame([input_data.features])
        
        # Ensure correct feature order
        expected_features = [
            'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
            'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 
            'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
        ]
        
        # Reorder features to match training
        features_ordered = features_df.reindex(columns=expected_features, fill_value=0)
        
        # Make prediction
        probability = float(model.predict_proba(features_ordered)[0, 1])
        
        return PredictionOutput(probability=probability)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/agent", response_model=AgentOutput)
async def agent_interaction(input_data: AgentInput):
    """Agentic interaction for credit risk analysis"""
    try:
        if credit_agent is None:
            raise HTTPException(status_code=503, detail="Credit agent not loaded")
        
        # Process the query through the agent
        response = credit_agent.process_query(input_data.features, input_data.query)
        
        return AgentOutput(**response)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Agent processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)