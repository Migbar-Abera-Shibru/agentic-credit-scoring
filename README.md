#  Agentic Credit Scoring 

A simple credit scoring system that predicts financial distress with explainable AI capabilities.

##  Features

- ** AI-Powered Predictions**: Random Forest model with 0.847 AUC
- ** Agentic Explanations**: Natural language understanding and reasoning
- ** RESTful API**: FastAPI with automatic documentation
- ** Risk Analysis**: Identifies specific risk factors and provides recommendations
- ** What-If Scenarios**: Simulate financial changes and their impact

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/agentic-credit-scoring.git
cd agentic-credit-scoring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (optional - pre-trained model included)
python main.py

# Start the API server
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# Check on Demo
python demo.py

```
### Usage

Access the API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)

**Basic Prediction:**

```
import requests

response = requests.post("http://localhost:8000/predict", json={
    "features": {
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 35,
        # ... other features
    }
})
```
### Agentic Analysis:

```
response = requests.post("http://localhost:8000/agent", json={
    "features": {
        "RevolvingUtilizationOfUnsecuredLines": 0.8,
        "age": 35,
        # ... other features
    },
    "query": "Explain my risk factors and suggest improvements"
})

```

Model Performance
-----------------

*   **AUC Score**: 0.847
    
*   **Target**: Serious Delinquency in 2 Years
    
*   **Features**: 10 financial and demographic variables




